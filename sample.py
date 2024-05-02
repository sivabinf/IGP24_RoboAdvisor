import pandas as pd
import numpy as np
import random

# Define the questions and their scoring options
questions = {
    "Investment Goals": {
        "Saving for retirement": 1,
        "Buying a property/car in the next 5-10 years": 2,
        "Generating short-term income": 3,
        "Speculative gains": 4,
        "Other": 0,
    },
    "Investment Time": {
        "Less than 1 year": 4,
        "1-3 years": 3,
        "3-5 years": 2,
        "More than 5 years": 1,
    },
    "Risk Tolerance": {
        "Sell all investments to avoid further losses": 1,
        "Sell some investments to minimize losses": 2,
        "Keep all investments and wait for the market to recover": 3,
        "Invest more to lower the average cost": 4,
    },
    "Option A or Option B": {
        "Investment A - I prefer stable and reasonable return with little risk, even if it mean missing out on higher returns": 2,
        "Investment B - I'm ready to take on increased risk for possible higher returns, even though I will experience more volatility and possibility of losses": 3,
    },
    "investing to buy a home within the next 10 years": {
        "Option A: I would prefer to maintain and preserve my initial investment": 2,
        "Option B: I will prefer maintain and preserving my initial investment": 3,
    },
    "In times of market volatility, how likely are you to make significant changes to your investment portfolio?": {
        "Very likely; I tend to react quickly to market changes": 4,
        "Somewhat likely; I might make some changes if the situation seems severe": 3,
        "Unlikely; I prefer to stick to my long-term investment plan": 1,
    },
    "How would you describe your experience with investing?": {
        "Beginner": 0,
        "Some experience": 0,
        "Experienced": 0,
        "Expert": 0,
    },
    "Which portfolio allocation would you be most comfortable with?": {
        "Mostly bonds and stable investments": 1,
        "A mix of stocks and bonds": 2,
        "Mostly stocks and high-growth investments": 3,
        "Speculative investments; cryptocurrencies, startups, etc": 4,
    },
    "How stable is your current source of income?": {
        "Very stable; My income is consistent and reliable": 4,
        "Moderately stable; My income varies somewhat but is generally predictable": 3,
        "Unstable. My income is highly variable or uncertain": 2,
        "Extremely unstable. I am in debt.": 1,
    },
    "Liquidity Preference": {
        "Very important. I prefer having immediate access": 1,
        "Somewhat important. I can wait a bit if needed": 2,
        "Not very important. I'm okay with locking in funds for higher returns for the short term": 3,
        "Not important at all. I am happy to lock in funds for high returns over the long term": 4,
    },
    "User preference. What are your investment preferences?": {
        "technology": 0,
        "oil & energy": 0,
        "Sustainable investment": 0,
        "Other": 0,
    },
}

# Number of profiles to generate
num_profiles = 1000

# Generate the dataset by picking a random choice for each question
data = []
for _ in range(num_profiles):
    profile = {}
    total_score = 0
    for question, options in questions.items():
        choice = random.choice(list(options.items()))
        profile[question] = choice[0]
        total_score += choice[1]
    profile['Total Score'] = total_score
    data.append(profile)

# Convert to DataFrame
df = pd.DataFrame(data)

# Calculate quintiles and assign risk categories
df['Risk Category'] = pd.qcut(df['Total Score'], 5, labels=["Very Low", "Low", "Moderate", "High", "Very High"])

# Show the first few rows of the dataset
df.head(), df['Risk Category'].value_counts()

# Saving the generated dataset to a CSV file -- from dataset4, data is updated 
csv_file_path = "Synthetic_Risk_Profile_Dataset_15.csv"
df.to_csv(csv_file_path, index=False)

csv_file_path
