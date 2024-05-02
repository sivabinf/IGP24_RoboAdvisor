import pandas as pd

# following the meeting on 30 April, I changed the values for three questions

# Load the dataset
df = pd.read_csv("Synthetic_Risk_Profile_Dataset_15.csv") 

# Show the first few rows of the dataset
print(df.head())  # Print the first five rows of the dataframe to check output

# Show value counts for the 'Risk Category' column
print(df['Risk Category'].value_counts())  # Print the distribution of values in the 'Risk Category' column
