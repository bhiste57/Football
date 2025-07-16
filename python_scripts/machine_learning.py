import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from python_scripts.support_functions import fetch_excel_database
import seaborn as sns


# --- 1. Load and prepare your data (from your original code) ---
df = fetch_excel_database()
df.sort_values(by='day', ignore_index=True, inplace=True)

# --- 2. Feature Engineering (from your original code) ---
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

    # Calculate average points won per game for the CURRENT match
    total_home_team_games = sum([team_history[home_team][k] for k in ['wins', 'draws', 'losses']])
    current_home_team_form = (team_history[home_team]['wins'] * 3 + team_history[home_team]['draws'] * 1) / (total_home_team_games if total_home_team_games > 0 else 1)

    total_away_team_games = sum([team_history[away_team][k] for k in ['wins', 'draws', 'losses']])
    current_away_team_form = (team_history[away_team]['wins'] * 3 + team_history[away_team]['draws'] * 1) / (total_away_team_games if total_away_team_games > 0 else 1)

    # Append to lists
    home_team_form_ratio_list.append(current_home_team_form)
    away_team_form_ratio_list.append(current_away_team_form)

    # Update team history for the NEXT match
    if row['result'] == 'H':
        team_history[home_team]['wins'] += 1
        team_history[away_team]['losses'] += 1
    elif row['result'] == 'D':
        team_history[home_team]['draws'] += 1
        team_history[away_team]['draws'] += 1
    else: # 'A'
        team_history[home_team]['losses'] += 1
        team_history[away_team]['wins'] += 1

# Add engineered features to the DataFrame
df['home_team_form_ratio'] = home_team_form_ratio_list
df['away_team_form_ratio'] = away_team_form_ratio_list

# Drop rows that have NaN values for features
df.dropna(inplace=True)

# Encode team names
le_home_team = LabelEncoder()
le_away_team = LabelEncoder()
df['home_team_encoded'] = le_home_team.fit_transform(df['home_team'])
df['away_team_encoded'] = le_away_team.fit_transform(df['away_team'])

# Define features
features = [
    'home_team_form_ratio',
    'away_team_form_ratio',
    'home_team_encoded',
    'away_team_encoded'
]



def train_model_up_to_day(df, target_day, features):
    """
    Train model only on data up to (but not including) the target day
    """
    # Only use data BEFORE the target day for training
    train_data = df[df['day'] < target_day].copy()

    if len(train_data) == 0:
        print(f"No training data available before day {target_day}")
        return None, None, None, None, None

    # Prepare features and target
    X_train = train_data[features]
    y_train = train_data['result']

    # Encode target
    le_result = LabelEncoder()
    y_train_encoded = le_result.fit_transform(y_train)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train model
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_clf.fit(X_train_scaled, y_train_encoded)

    print(f"Model trained on {len(train_data)} matches from days 1-{target_day - 1}")

    return rf_clf, scaler, le_result, train_data, X_train_scaled


def predict_day_realistically(df, target_day, features):
    """
    Predict results for a specific day using only historical data
    """
    # Train model on data before target day
    model, scaler, le_result, train_data, X_train_scaled = train_model_up_to_day(df, target_day, features)

    if model is None:
        return None

    # Get matches for the target day
    target_matches = df[df['day'] == target_day].copy()

    if len(target_matches) == 0:
        print(f"No matches found for day {target_day}")
        return None

    # Make predictions
    X_test = target_matches[features]
    X_test_scaled = scaler.transform(X_test)

    # Get predictions and probabilities
    predictions_encoded = model.predict(X_test_scaled)
    predictions = le_result.inverse_transform(predictions_encoded)
    probabilities = model.predict_proba(X_test_scaled)

    # Add predictions to dataframe
    target_matches['predicted'] = predictions

    # Add probability columns
    for i, class_label in enumerate(le_result.classes_):
        target_matches[f'prob_{class_label}'] = probabilities[:, i]

    # Calculate accuracy
    accuracy = accuracy_score(target_matches['result'], predictions)

    # Print results
    print(f"\n=== Day {target_day} Realistic Predictions ===")
    print(f"Training data: Days 1-{target_day - 1} ({len(train_data)} matches)")
    print(f"Test data: Day {target_day} ({len(target_matches)} matches)")
    print(f"Accuracy: {accuracy:.2%}")

    return target_matches, model, scaler, le_result


def visualize_realistic_predictions(target_matches, target_day):
    """
    Visualize realistic predictions vs actual results
    """
    if target_matches is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Day {target_day} Realistic Predictions vs Actual Results', fontsize=16, fontweight='bold')

    # 1. Match-by-match comparison
    ax1 = axes[0, 0]
    matches_labels = [f"{row['home_team'][:3]} vs {row['away_team'][:3]}" for _, row in target_matches.iterrows()]
    x_pos = np.arange(len(matches_labels))

    color_map = {'H': 'green', 'A': 'red', 'D': 'blue'}
    actual_colors = [color_map[result] for result in target_matches['result']]
    pred_colors = [color_map[pred] for pred in target_matches['predicted']]

    width = 0.35
    ax1.bar(x_pos - width / 2, [1] * len(matches_labels), width, label='Actual', color=actual_colors, alpha=0.7)
    ax1.bar(x_pos + width / 2, [1] * len(matches_labels), width, label='Predicted', color=pred_colors, alpha=0.7)

    ax1.set_xlabel('Matches')
    ax1.set_ylabel('Results')
    ax1.set_title('Match-by-Match Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(matches_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1.2)

    # Add result labels
    for i, (actual, pred) in enumerate(zip(target_matches['result'], target_matches['predicted'])):
        ax1.text(i - width / 2, 0.5, actual, ha='center', va='center', fontweight='bold')
        ax1.text(i + width / 2, 0.5, pred, ha='center', va='center', fontweight='bold')

    # 2. Accuracy pie chart
    ax2 = axes[0, 1]
    correct = sum(target_matches['result'] == target_matches['predicted'])
    total = len(target_matches)

    ax2.pie([correct, total - correct],
            labels=[f'Correct ({correct})', f'Wrong ({total - correct})'],
            autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
    ax2.set_title(f'Prediction Accuracy: {correct / total:.2%}')

    # 3. Confidence distribution
    ax3 = axes[1, 0]

    # Get max probability for each prediction (confidence)
    prob_cols = [col for col in target_matches.columns if col.startswith('prob_')]
    confidences = target_matches[prob_cols].max(axis=1)

    ax3.hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_xlabel('Prediction Confidence')
    ax3.set_ylabel('Number of Matches')
    ax3.set_title('Distribution of Prediction Confidence')
    ax3.axvline(confidences.mean(), color='red', linestyle='--', label=f'Mean: {confidences.mean():.3f}')
    ax3.legend()

    # 4. Detailed results table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')

    table_data = []
    for idx, row in target_matches.iterrows():
        confidence = max([row[col] for col in prob_cols])
        correct_symbol = '✓' if row['result'] == row['predicted'] else '✗'
        table_data.append([
            f"{row['home_team'][:8]} vs {row['away_team'][:8]}",
            f"{row['home_score']}-{row['away_score']}",
            row['result'],
            row['predicted'],
            correct_symbol,
            f"{confidence:.3f}"
        ])

    table = ax4.table(cellText=table_data,
                      colLabels=['Match', 'Score', 'Actual', 'Predicted', 'Correct', 'Confidence'],
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title('Detailed Results')

    plt.tight_layout()
    plt.show()

    # Print detailed results
    print(f"\n=== Match Details ===")
    for idx, row in target_matches.iterrows():
        confidence = max([row[col] for col in prob_cols])
        status = "✓" if row['result'] == row['predicted'] else "✗"
        print(f"{row['home_team']} vs {row['away_team']}: {row['home_score']}-{row['away_score']} "
              f"(Actual: {row['result']}, Predicted: {row['predicted']}, "
              f"Confidence: {confidence:.3f}) {status}")


def evaluate_multiple_days(df, features, start_day=15, end_day=25):
    """
    Evaluate model performance across multiple days
    """
    results = []

    for day in range(start_day, end_day + 1):
        target_matches, model, scaler, le_result = predict_day_realistically(df, day, features)

        if target_matches is not None:
            accuracy = accuracy_score(target_matches['result'], target_matches['predicted'])
            results.append({
                'day': day,
                'accuracy': accuracy,
                'num_matches': len(target_matches),
                'correct_predictions': sum(target_matches['result'] == target_matches['predicted'])
            })

    results_df = pd.DataFrame(results)

    # Plot performance over time
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['day'], results_df['accuracy'], marker='o', linewidth=2, markersize=8)
    plt.xlabel('Day')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Over Time (Realistic Evaluation)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=results_df['accuracy'].mean(), color='red', linestyle='--',
                label=f'Average: {results_df["accuracy"].mean():.2%}')
    plt.legend()
    plt.show()

    print(f"\n=== Overall Performance Summary ===")
    print(f"Days evaluated: {start_day}-{end_day}")
    print(f"Average accuracy: {results_df['accuracy'].mean():.2%}")
    print(
        f"Best day: Day {results_df.loc[results_df['accuracy'].idxmax(), 'day']} ({results_df['accuracy'].max():.2%})")
    print(
        f"Worst day: Day {results_df.loc[results_df['accuracy'].idxmin(), 'day']} ({results_df['accuracy'].min():.2%})")

    return results_df


# Usage examples:
print("=== REALISTIC EVALUATION ===")

# 1. Predict day 15 using only data from days 1-14
day_15_results, model, scaler, le_result = predict_day_realistically(df, 15, features)
if day_15_results is not None:
    visualize_realistic_predictions(day_15_results, 15)

# 2. Evaluate multiple days
print("\n" + "=" * 50)
performance_df = evaluate_multiple_days(df, features, start_day=15, end_day=25)