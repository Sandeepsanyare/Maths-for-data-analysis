import numpy as np
import pandas as pd

# Step 1: Create the dataset with 20,000 data points
np.random.seed(0)  # For reproducibility

# Define age groups and weights
age_groups = list(range(25, 66, 5))  # [25, 30, 35, ..., 65]
purchase_weights = [0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.7, 0.85, 0.9]

# Generate random data
ages = np.random.choice(age_groups, size=20000)
purchases = (np.random.rand(20000) < np.array([purchase_weights[age_groups.index(age)] for age in ages])).astype(int)

# Create DataFrame
data = pd.DataFrame({'Age': ages, 'Purchase': purchases})

# Display sample data
print(data.head())

# Step 2: Assign uniform probabilities (50% chance for all ages)
data['Purchase_No_Weight'] = np.random.binomial(1, 0.5, size=20000)

# Display sample data
print(data.head())

# Calculate conditional probability with weighted probabilities
print("\nConditional Probabilities with Weighted Probabilities:")

for age in age_groups:
    # Filter data for the specific age group
    age_data = data[data['Age'] == age]
    
    # Calculate P(Purchase | Age=age) with weights
    p_given_age_weighted = age_data['Purchase'].mean()
    
    # Print the result
    print(f"Age {age}: P(Purchase | Age={age}) with weight = {p_given_age_weighted:.4f}")

    # Calculate conditional probability without weighted probabilities
print("\nConditional Probabilities with Uniform Probabilities:")

for age in age_groups:
    # Filter data for the specific age group
    age_data_no_weight = data[data['Age'] == age]
    
    # Calculate P(Purchase | Age=age) without weights
    p_given_age_no_weight = age_data_no_weight['Purchase_No_Weight'].mean()
    
    # Print the result
    print(f"Age {age}: P(Purchase | Age={age}) without weight = {p_given_age_no_weight:.4f}")

