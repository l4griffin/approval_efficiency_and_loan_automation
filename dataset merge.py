import pandas as pd

# 1. Load the attached dataset
df_train = pd.read_csv('Train.csv')

# 2. Load the previously generated synthetic data
df_synth = pd.read_csv('african_context_digital_credit_data.csv')

# 3. Perform an index-based join
# This preserves all rows of Train.csv 
df_merged = df_train.join(df_synth)

# 4. Save the merged output to a new file
df_merged.to_csv('Merged_Dataset.csv', index=False)

print(f"Original Train.csv shape: {df_train.shape}")
print(f"Merged dataset shape: {df_merged.shape}")
