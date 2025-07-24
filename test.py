import pandas as pd

# Replace the path below with the actual path to your GSM8K_Zulu parquet file
parquet_file = 'GSM8K_Zulu.parquet'

# Load the dataset
df = pd.read_parquet(parquet_file)

# Display the first few rows
print(df.head())
