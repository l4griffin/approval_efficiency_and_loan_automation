import pandas as pd
import numpy as np

# Define the number of records needed
n_records = 5000
np.random.seed(101)

# Generate synthetic attributes based on typical digital lending profiles
data = {
    'Monthly_Income': np.round(np.random.normal(45000, 15000, n_records).clip(10000, 200000), 0),
    'Mobile_Money_Tx_Volume': np.round(np.random.normal(60000, 25000, n_records).clip(5000, 500000), 0),
    'Past_Defaults': np.random.choice([0, 1, 2, 3], size=n_records, p=[0.75, 0.15, 0.07, 0.03])
}

df = pd.DataFrame(data)

# Calculate Loan Amount as a realistic fraction of Monthly Income
df['Loan_Amount'] = np.round((df['Monthly_Income'] * np.random.uniform(0.1, 0.6, n_records)).clip(500, 50000), 0)

# Determine the Target Variable (Default_Status)
# Risk increases with past defaults and high loan-to-income ratios, but decreases with high transaction volumes
loan_to_income = df['Loan_Amount'] / df['Monthly_Income']
risk_score = (df['Past_Defaults'] * 0.35) + (loan_to_income * 0.4) - (df['Mobile_Money_Tx_Volume'] / 500000)

# Add Gaussian noise to introduce real-world unpredictability into the data
risk_score += np.random.normal(0, 0.2, n_records)

# Classify as 1 (Default) if risk score exceeds a threshold, otherwise 0 (Repaid)
df['Default_Status'] = np.where(risk_score > 0.4, 1, 0)

# Reorder columns to match the exact requirement
df = df[['Monthly_Income', 'Loan_Amount', 'Mobile_Money_Tx_Volume', 'Past_Defaults', 'Default_Status']]

# Export the final dataset to a CSV file
df.to_csv('african_context_digital_credit_data.csv', index=False)
print("Dataset successfully generated and saved.")



