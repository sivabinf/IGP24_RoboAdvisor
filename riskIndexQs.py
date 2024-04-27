import uuid
import csv
import math

def collect_responses_and_calculate_score(questions):
    total_score = 0
    responses = {}
    for question, options in questions.items():
        print(f"\n{question}:")
        option_list = list(options.keys())
        for i, option in enumerate(option_list, 1):
            print(f"{i}. {option}")

        while True:
            try:
                response = input("Your choice (number): ")
                response_index = int(response) - 1
                if response_index < 0 or response_index >= len(option_list):
                    print("Invalid choice. Please enter a number from the list.")
                else:
                    chosen_option = option_list[response_index]
                    total_score += options[chosen_option]
                    responses[question] = chosen_option  # Store the user's choice
                    break
            except ValueError:
                print("Invalid input. Please enter a valid number.")
    return total_score, responses

# Calculate the risk score and capture responses
# Questionnaire
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
        "Investment A - I prefer stable and reasonable return with little risk, even if it mean missing out on higher returns ": 1,
        "Investment B - I'm ready to take on increased risk for possible higher returns, even though I will experience more volatility and possibility of losses ": 4,
    },
    "investing to buy a home within the next 10 years": {
        "Option A: I would prefer to maintain and preserve my initial investment": 1,
        "Option A: I will prefer maintain and preserving my initial investment": 4,
    },
    "In times of market volatility, how likely are you to make significant changes to your investment portfolio?": {
        "Very likely; I tend to react quickly to market changes": 4,
        "Somewhat likely; I might make some changes if the situation seems severe ": 3,
        "Unlikely; I prefer to stick to my long-term investment plan ": 1,
    },
    "How would you describe your experience with investing?": {
        "Beginner": 1,
        "Some experience": 2,
        "Experienced": 3,
        "Expert": 4,
    },
    "Which portfolio allocation would you be most comfortable with?": {
        "Mostly bonds and stable investments": 1,
        "A mix of stocks and bonds": 2,
        "Mostly stocks and high-growth investments": 3,
        "Speculative investments; cryptocurrencies, startups, etc ": 4,
    },
    "How stable is your current source of income?": {
        "Very stable; My income is consistent and reliable": 4,
        "Moderately stable; My income varies somewhat but is generally predictable": 3,
        "Unstable. My income is highly variable or uncertain ": 2,
        "Extremely unstable. I am in debt.": 1,
    },
    "Liquidity Preference": {
        "Very important. I prefer having immediate access": 1,
        "Somewhat important. I can wait a bit if needed ": 2,
        "Not very important. I'm okay with locking in funds for higher returns for the short term": 3,
        "Not important at all. I am happy to lock in funds for high returns over the long term": 4,
    },
    "User preference. What are your investment preferences?": {
        "technology": 0,
        "oil & energy ": 0,
        "Sustainable investment": 0,
        "Other": 0,
    },

}
max_possible_score = sum(max(options.values()) for options in questions.values())
risk_score, user_responses = collect_responses_and_calculate_score(questions)
risk_percentage = risk_score / max_possible_score  # This gives a value from 0.0 to just below 1.0

# Convert risk percentage to a decile index
risk_decile = min(math.floor(risk_percentage * 10) + 1, 10)

user_id = uuid.uuid4().hex
investment_preference = user_responses.get("User preference. What are your investment preferences?", "No selection made")

# Write to CSV
with open('user_data.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write header if the file is newly created
    if file.tell() == 0:
        writer.writerow(["User ID", "Risk Score", "Risk Decile", "Investment Preference"])
    writer.writerow([user_id, f"{risk_percentage:.2f}", risk_decile, investment_preference])

print(f"Data saved for User ID: {user_id}")

# Print the results
print(f"\nUser ID: {user_id}")
print(f"Total Risk Score: {risk_score}")
print(f"Risk Percentage: {risk_percentage:.2f}%")
print(f"Risk Decile: {risk_decile}")
print(f"Investment Preference: {investment_preference}")
