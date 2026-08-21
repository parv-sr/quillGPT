import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ============================================================
# Configuration
# ============================================================

SEED = 42

N = 3
MATRIX_SIZE = N * (N + 1)

HIDDEN_SIZE = 256

LEARNING_RATE = 3e-4
GAMMA = 0.99

EPISODES = 100_000
MAX_STEPS = 50

PRINT_EVERY = 1_000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Actions
# ============================================================

@dataclass(frozen=True)
class Action:
    name: str
    operation: str
    row_a: int
    row_b: int | None = None


def build_action_space() -> list[Action]:

    actions = []

    # --------------------------------------------------------
    # Row swaps
    # --------------------------------------------------------

    for i in range(N):
        for j in range(i + 1, N):
            actions.append(
                Action(
                    name=f"SWAP R{i} R{j}",
                    operation="swap",
                    row_a=i,
                    row_b=j,
                )
            )

    # --------------------------------------------------------
    # Row normalization
    # --------------------------------------------------------

    for i in range(N):
        actions.append(
            Action(
                name=f"NORMALIZE R{i}",
                operation="normalize",
                row_a=i,
            )
        )

    # --------------------------------------------------------
    # Row elimination
    #
    # ELIMINATE R_i USING R_j
    #
    # The environment determines the coefficient.
    # --------------------------------------------------------

    for i in range(N):
        for j in range(N):

            if i == j:
                continue

            actions.append(
                Action(
                    name=f"ELIMINATE R{i} USING R{j}",
                    operation="eliminate",
                    row_a=i,
                    row_b=j,
                )
            )

    return actions


ACTIONS = build_action_space()
N_ACTIONS = len(ACTIONS)

print(f"Device: {DEVICE}")
print(f"Number of actions: {N_ACTIONS}")

for i, action in enumerate(ACTIONS):
    print(f"{i:2d}: {action.name}")


# ============================================================
# Gaussian Elimination Environment
# ============================================================

class GaussianEliminationEnvironment:

    def __init__(
        self,
        n: int = 3,
        max_steps: int = 50,
    ):

        self.n = n
        self.max_steps = max_steps

        self.matrix = None
        self.steps = 0

    # --------------------------------------------------------
    # Generate a random invertible matrix
    # --------------------------------------------------------

    def _generate_system(self):

        while True:

            A = np.random.uniform(
                -2.0,
                2.0,
                size=(self.n, self.n),
            )

            determinant = np.linalg.det(A)

            if abs(determinant) > 0.25:
                break

        b = np.random.uniform(
            -2.0,
            2.0,
            size=(self.n, 1),
        )

        return np.concatenate([A, b], axis=1)

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(self):

        self.matrix = self._generate_system()
        self.steps = 0

        return self._state()

    # --------------------------------------------------------
    # State representation
    # --------------------------------------------------------

    def _state(self):

        state = self.matrix.flatten().astype(np.float32)

        return torch.tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        )

    # --------------------------------------------------------
    # Distance from identity
    # --------------------------------------------------------

    def identity_distance(self):

        A = self.matrix[:, :self.n]

        I = np.eye(self.n)

        return np.mean(
            np.abs(A - I)
        )

    # --------------------------------------------------------
    # Is the coefficient matrix identity?
    # --------------------------------------------------------

    def is_solved(self):

        A = self.matrix[:, :self.n]

        I = np.eye(self.n)

        return np.allclose(
            A,
            I,
            atol=1e-3,
        )

    # --------------------------------------------------------
    # Apply action
    # --------------------------------------------------------

    def step(self, action_index: int):

        action = ACTIONS[action_index]

        old_distance = self.identity_distance()

        valid = True

        # ----------------------------------------------------
        # SWAP
        # ----------------------------------------------------

        if action.operation == "swap":

            i = action.row_a
            j = action.row_b

            self.matrix[[i, j]] = self.matrix[[j, i]]

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        elif action.operation == "normalize":

            i = action.row_a

            pivot = self.matrix[i, i]

            if abs(pivot) < 1e-8:

                valid = False

            else:

                self.matrix[i] /= pivot

        # ----------------------------------------------------
        # ELIMINATE
        # ----------------------------------------------------

        elif action.operation == "eliminate":

            target = action.row_a
            source = action.row_b

            # ------------------------------------------------
            # Determine which pivot column to use.
            #
            # We select the diagonal position associated
            # with the source row.
            # ------------------------------------------------

            pivot_column = source

            source_value = self.matrix[source, pivot_column]
            target_value = self.matrix[target, pivot_column]

            if abs(source_value) < 1e-8:

                valid = False

            else:

                coefficient = (
                    -target_value / source_value
                )

                self.matrix[target] += (
                    coefficient * self.matrix[source]
                )

        self.steps += 1

        new_distance = self.identity_distance()

        # ====================================================
        # Reward
        # ====================================================

        if not valid:

            reward = -2.0

        else:

            progress = old_distance - new_distance

            reward = progress * 10.0

            # Small penalty for taking another step
            reward -= 0.01

        # ----------------------------------------------------
        # Terminal reward
        # ----------------------------------------------------

        solved = self.is_solved()

        if solved:

            reward += 100.0

        done = (
            solved
            or self.steps >= self.max_steps
        )

        return (
            self._state(),
            reward,
            done,
            {
                "solved": solved,
                "valid": valid,
                "distance": new_distance,
            },
        )


# ============================================================
# Policy Network
# ============================================================

class PolicyNetwork(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                hidden_size,
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_size,
                hidden_size,
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_size,
                hidden_size,
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_size,
                output_size,
            ),
        )

    def forward(self, x):

        logits = self.network(x)

        return logits


# ============================================================
# Policy Gradient Agent
# ============================================================

class Agent:

    def __init__(self):

        self.model = PolicyNetwork(
            input_size=MATRIX_SIZE,
            hidden_size=HIDDEN_SIZE,
            output_size=N_ACTIONS,
        ).to(DEVICE)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE,
        )

    # --------------------------------------------------------
    # Choose action
    # --------------------------------------------------------

    def choose_action(self, state):

        logits = self.model(state)

        distribution = Categorical(
            logits=logits
        )

        action = distribution.sample()

        log_probability = distribution.log_prob(
            action
        )

        return (
            action.item(),
            log_probability,
        )

    # --------------------------------------------------------
    # Update policy
    # --------------------------------------------------------

    def update(
        self,
        log_probabilities,
        rewards,
    ):

        returns = []

        discounted = 0.0

        for reward in reversed(rewards):

            discounted = (
                reward
                + GAMMA * discounted
            )

            returns.insert(
                0,
                discounted,
            )

        returns = torch.tensor(
            returns,
            dtype=torch.float32,
            device=DEVICE,
        )

        # ----------------------------------------------------
        # Normalize returns.
        #
        # This dramatically improves stability.
        # ----------------------------------------------------

        if len(returns) > 1:

            returns = (
                returns - returns.mean()
            ) / (
                returns.std() + 1e-8
            )

        loss = 0.0

        for log_probability, return_value in zip(
            log_probabilities,
            returns,
        ):

            loss += (
                -log_probability
                * return_value
            )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0,
        )

        self.optimizer.step()

        return loss.item()


# ============================================================
# Training
# ============================================================

def train():

    environment = GaussianEliminationEnvironment(
        n=N,
        max_steps=MAX_STEPS,
    )

    agent = Agent()

    solved_count = 0

    reward_history = []

    for episode in range(1, EPISODES + 1):

        state = environment.reset()

        log_probabilities = []
        rewards = []

        total_reward = 0.0

        for step in range(MAX_STEPS):

            action, log_probability = (
                agent.choose_action(state)
            )

            (
                next_state,
                reward,
                done,
                info,
            ) = environment.step(action)

            log_probabilities.append(
                log_probability
            )

            rewards.append(reward)

            total_reward += reward

            state = next_state

            if done:

                if info["solved"]:
                    solved_count += 1

                break

        loss = agent.update(
            log_probabilities,
            rewards,
        )

        reward_history.append(
            total_reward
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if episode % PRINT_EVERY == 0:

            recent_rewards = reward_history[
                -PRINT_EVERY:
            ]

            average_reward = np.mean(
                recent_rewards
            )

            success_rate = (
                solved_count / PRINT_EVERY
            )

            print(
                f"\nEpisode {episode:,}"
            )

            print(
                f"Average reward: "
                f"{average_reward:.3f}"
            )

            print(
                f"Success rate: "
                f"{success_rate:.2%}"
            )

            print(
                f"Last loss: "
                f"{loss:.5f}"
            )

            solved_count = 0

    return agent


# ============================================================
# Evaluation
# ============================================================

def evaluate(agent, episodes=20):

    environment = GaussianEliminationEnvironment(
        n=N,
        max_steps=MAX_STEPS,
    )

    agent.model.eval()

    successes = 0

    for episode in range(episodes):

        state = environment.reset()

        print("\n" + "=" * 60)

        print(
            f"Episode {episode + 1}"
        )

        print("\nInitial matrix:")

        print(
            environment.matrix
        )

        print()

        for step in range(MAX_STEPS):

            with torch.no_grad():

                logits = agent.model(state)

                action = torch.argmax(
                    logits
                ).item()

            action_info = ACTIONS[action]

            print(
                f"Step {step + 1:2d}: "
                f"{action_info.name}"
            )

            (
                state,
                reward,
                done,
                info,
            ) = environment.step(action)

            print(
                environment.matrix
            )

            print(
                f"Distance: "
                f"{info['distance']:.6f}"
            )

            if done:

                if info["solved"]:

                    print(
                        "\nSOLVED!"
                    )

                    successes += 1

                else:

                    print(
                        "\nFAILED."
                    )

                break

    print("\n" + "=" * 60)

    print(
        f"Evaluation success rate: "
        f"{successes / episodes:.2%}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    agent = train()

    torch.save(
        agent.model.state_dict(),
        "gauss_jordan_policy.pt",
    )

    evaluate(
        agent,
        episodes=10,
    )