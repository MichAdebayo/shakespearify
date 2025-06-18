import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("garnavaurha/shakespearify")

print("Path to dataset files:", path)

# Load the dataset into a pandas DataFrame
df = pd.read_csv(f"{path}/final.csv")

print("Dataset loaded successfully!")

print("First few rows of the dataset:")
print(df.head())