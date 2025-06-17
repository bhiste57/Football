import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from python_scripts.support_functions import fetch_excel_database


# --- 1. Load your data ---
df = fetch_excel_database()

# Sort by day to ensure correct chronological feature engineering
df.sort_values(by='day', ignore_index=True, inplace=True)


# --- 2. Feature Engineering ---

# Create Result (Target Variable)
def get_result(row: pd.Series) -> str:
    if row['home_score'] > row['away_score']:
        return 'H' # Home Win
    elif row['home_score'] < row['away_score']:
        return 'A' # Away Win
    else:
        return 'D' # Draw
df['result'] = df.apply(get_result, axis=1)

# Initialize dictionaries to store team statistics up to the current match
team_history = {team: {'wins': 0, 'draws': 0, 'losses': 0}
                for team in pd.concat([df['home_team'], df['away_team']]).unique()}

# Lists to collect features for each match
home_team_form_ratio_list = []
away_team_form_ratio_list = []

# Iterate through matches chronologically to calculate and collect features
for i, row in df.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']

    # --- Calculate average points won per game for the CURRENT match ---
    total_home_team_games = sum([team_history[home_team][k] for k in ['wins', 'draws', 'losses']]) # number of games played by the home team
    current_home_team_form = (team_history[home_team]['wins'] * 3 + team_history[home_team]['draws'] * 1) / (total_home_team_games if total_home_team_games > 0 else 1)

    total_away_team_games = sum([team_history[away_team][k] for k in ['wins', 'draws', 'losses']]) # number of games played by the away team
    current_away_team_form = (team_history[away_team]['wins'] * 3 + team_history[away_team]['draws'] * 1) / (total_away_team_games if total_away_team_games > 0 else 1)

    # Append to lists
    home_team_form_ratio_list.append(current_home_team_form)
    away_team_form_ratio_list.append(current_away_team_form)

    # --- Update team history for the NEXT match (after this match's result is known) ---
    if row['result'] == 'H':
        team_history[home_team]['wins'] += 1
        team_history[away_team]['losses'] += 1
    elif row['result'] == 'D':
        team_history[home_team]['draws'] += 1
        team_history[away_team]['draws'] += 1
    else: # 'A'
        team_history[home_team]['losses'] += 1
        team_history[away_team]['wins'] += 1

# Add engineered features to the DataFrame # should be matching the teams since we re iterating on the rows before
df['home_team_form_ratio'] = home_team_form_ratio_list
df['away_team_form_ratio'] = away_team_form_ratio_list

# Drop rows that have NaN values for features (e.g., very first few games with no history)
df.dropna(inplace=True)

# Encode team names
le_home_team = LabelEncoder()
le_away_team = LabelEncoder()
df['home_team_encoded'] = le_home_team.fit_transform(df['home_team'])
df['away_team_encoded'] = le_away_team.fit_transform(df['away_team'])

# Features (X) and Target (y) - SIMPLIFIED FEATURES
features = [
    'home_team_form_ratio',
    'away_team_form_ratio',
    'home_team_encoded',
    'away_team_encoded']
X = df[features]
y = df['result']

# Encode the target variable (H, D, A to numbers)
le_result = LabelEncoder()
y_encoded = le_result.fit_transform(y)

# --- 3. Train-Test Split (Time-based split is crucial for realistic evaluation) ---
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.01, random_state=42, stratify=y_encoded)
# Using a very small test_size here because the dummy data is small.
# For 38 rounds, use a time-based split: X_train = df[df['day'] <= split_date][features] etc.

# Scale numerical features (even just form points benefit from scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. Model Training: Random Forest Classifier ---
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_clf.fit(X_train_scaled, y_train)

# --- Evaluate the model (optional, but good practice) ---
y_pred = rf_clf.predict(X_test_scaled)
print("Model Evaluation on Test Set (Simplified Features):")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le_result.classes_))

# --- 5. Prediction Function for New Games ---

def predict_match_outcome_simple(home_team_name, away_team_name, model, scaler, team_history_data,
                                  le_home, le_away, le_result_encoder, features_list):
    """
    Predicts the outcome (H, A, D) and probabilities for a single new match using simplified features.

    Args:
        home_team_name (str): Name of the home team.
        away_team_name (str): Name of the away team.
        model: Trained scikit-learn model (e.g., RandomForestClassifier).
        scaler: Trained StandardScaler used during training.
        team_history_data (dict): Dictionary containing the latest calculated form points for all teams.
        le_home (LabelEncoder): LabelEncoder fitted on home team names.
        le_away (LabelEncoder): LabelEncoder fitted on away team names.
        le_result_encoder (LabelEncoder): LabelEncoder fitted on match results (H, A, D).
        features_list (list): List of feature names used during training.

    Returns:
        tuple: (predicted_outcome_label, probabilities_dict)
    """
    home_team_stats = team_history_data.get(home_team_name, None)
    away_team_stats = team_history_data.get(away_team_name, None)

    if home_team_stats is None:
        print(f"Warning: No historical data found for Home Team '{home_team_name}'. Using default 0s.")
        home_team_stats = {'wins': 0, 'draws': 0, 'losses': 0}
    if away_team_stats is None:
        print(f"Warning: No historical data found for Away Team '{away_team_name}'. Using default 0s.")
        away_team_stats = {'wins': 0, 'draws': 0, 'losses': 0}

    total_home_games = sum([home_team_stats[k] for k in ['wins', 'draws', 'losses']])
    current_home_form = (home_team_stats['wins'] * 3 + home_team_stats['draws'] * 1) / (total_home_games if total_home_games > 0 else 1)

    total_away_games = sum([away_team_stats[k] for k in ['wins', 'draws', 'losses']])
    current_away_form = (away_team_stats['wins'] * 3 + away_team_stats['draws'] * 1) / (total_away_games if total_away_games > 0 else 1)

    try:
        encoded_home_team = le_home.transform([home_team_name])[0]
    except ValueError:
        print(f"Warning: Home team '{home_team_name}' not seen during training. Assigning default encoding.")
        encoded_home_team = 0
    try:
        encoded_away_team = le_away.transform([away_team_name])[0]
    except ValueError:
        print(f"Warning: Away team '{away_team_name}' not seen during training. Assigning default encoding.")
        encoded_away_team = 0

    new_match_df = pd.DataFrame([{
        'home_form_points': current_home_form,
        'away_form_points': current_away_form,
        'home_team_encoded': encoded_home_team,
        'away_team_encoded': encoded_away_team
    }], columns=features_list)

    new_match_scaled = scaler.transform(new_match_df)

    predicted_encoded = model.predict(new_match_scaled)[0]
    predicted_outcome_label = le_result_encoder.inverse_transform([predicted_encoded])[0]

    probabilities = model.predict_proba(new_match_scaled)[0]
    probabilities_dict = {}
    for i, class_label in enumerate(le_result_encoder.classes_):
        probabilities_dict[class_label] = probabilities[i]

    return predicted_outcome_label, probabilities_dict

# --- Get the final team_history for prediction after processing all historical data ---
# This dictionary 'team_history' now holds the up-to-date stats (form points only) for all teams.
final_team_form_stats = {}
for team, stats in team_history.items():
    total_games = sum([stats[k] for k in ['wins', 'draws', 'losses']])
    form = (stats['wins'] * 3 + stats['draws'] * 1) / (total_games if total_games > 0 else 1)
    final_team_form_stats[team] = {'wins': stats['wins'], 'draws': stats['draws'], 'losses': stats['losses'], 'form_points': form}


# --- Example Usage for a "Next Round" Game ---
print("\n--- Predicting for a hypothetical next game (Simplified Model) ---")
next_home_team = 'Team A' # Replace with actual team names from your league
next_away_team = 'Team F'

predicted_outcome, probabilities = predict_match_outcome_simple(
    next_home_team, next_away_team, rf_clf, scaler, team_history, # Pass team_history directly, as predict_match_outcome_simple needs individual win/draw/loss counts
    le_home_team, le_away_team, le_result, features
)

print(f"\nMatch: {next_home_team} (Home) vs {next_away_team} (Away)")
print(f"Predicted Outcome: {predicted_outcome}")
print("Probabilities:")
for outcome, prob in probabilities.items():
    print(f"  {outcome}: {prob:.4f}")