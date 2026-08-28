import os
from datasets import load_dataset

# 1. Define your desired path and filename
# Example for Windows: r"C:\Users\YourName\Documents\ML_Project\1800s_corpus.txt"
# Example for Mac/Linux: "/Users/YourName/Documents/ML_Project/1800s_corpus.txt"
desired_path = r"C:\F DRIVE\tinyGPT\data\raw\input_10.txt"

# 2. Automatically create the folders if they don't exist yet
output_dir = os.path.dirname(desired_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 3. Stream the dataset and write to your path
print(f"Downloading and saving text to: {desired_path}")
dataset = load_dataset("deepmind/pg19", split="validation", streaming=True)

with open(desired_path, "w", encoding="utf-8") as f:
    for i, row in enumerate(dataset):
        if row['text']:  # Ensure the text field isn't empty
            f.write(row['text'] + "\n\n")
        
        # Status update every 100 books
        if i % 100 == 0 and i > 0:
            print(f"Processed {i} books...")
            
        # Target token size check: 
        # For a 27M param model, aiming for ~500 books will yield ~50MB–100MB of data.
        if i >= 500:  
            break

print("Done! File saved successfully.")
