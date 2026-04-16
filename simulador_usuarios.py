import random
import pandas as pd

# Constants for simulation
NUM_USERS = 1000
DIFFICULTY_THRESHOLD = 65

# Define user profiles
profiles = ['Newcomer', 'Occasional User', 'Regular User', 'Power User']

# Define behavioral factors
behavioral_factors = {
    'user_experience': [0, 1, 2, 3],  # 0: Low, 3: High
    'interactions': list(range(1, 101)),  # 1 to 100 interactions
    'accessibility_issues': [0, 1, 2]  # 0: None, 2: Many
}

# Function to calculate difficulty score
def calculate_difficulty_score(user):
    # Weighted factors
    weights = {
        'profile': 0.3,
        'interactions': 0.5,
        'accessibility_issues': 0.2,
    }
    return (weights['profile'] * user['profile_weight'] +
            weights['interactions'] * user['interactions'] +
            weights['accessibility_issues'] * user['accessibility_issues']) * 100 / (sum(weights.values()))

# Function to simulate user churn
def predict_churn(users):
    for user in users:
        if user['difficulty_score'] > DIFFICULTY_THRESHOLD:
            user['churn'] = 1  # Churn
        else:
            user['churn'] = 0  # Not churn

# Generate synthetic user data
users = []
for _ in range(NUM_USERS):
    profile = random.choice(profiles)
    profile_weight = profiles.index(profile)  # Assign weights based on profile type
    interactions = random.choice(behavioral_factors['interactions'])
    accessibility_issues = random.choice(behavioral_factors['accessibility_issues'])
    difficulty_score = calculate_difficulty_score({
        'profile_weight': profile_weight,
        'interactions': interactions,
        'accessibility_issues': accessibility_issues,
    })

    users.append({
        'profile': profile,
        'interactions': interactions,
        'accessibility_issues': accessibility_issues,
        'difficulty_score': difficulty_score,
        'churn': 0  # Default to not churned
    })

# Predict churn based on difficulty score
predict_churn(users)

# Save to DataFrame and export as CSV
users_df = pd.DataFrame(users)
users_df.to_csv('synthetic_user_data.csv', index=False)
print('Synthetic user data generated and saved to synthetic_user_data.csv')
