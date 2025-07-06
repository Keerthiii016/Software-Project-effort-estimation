'''import joblib
import pandas as pd
import numpy as np

# Load the trained model and selected features
model = joblib.load("rf_model.pkl")
selected_features = joblib.load("features.pkl")


def predict_effort(user_input):

    try:
        # Convert user input into a DataFrame
        input_df = pd.DataFrame([user_input])

        # Ensure only selected features are used
        input_df = input_df[selected_features]
       #input_df = input_df[[col for col in selected_features if col in input_df.columns]  ''''''

        # Predict effort using the trained model
        effort_prediction = model.predict(input_df)[0]

        return round(effort_prediction, 2)

    except Exception as e:
        return f"Prediction Error: {str(e)}"
'''

'''import joblib
import pandas as pd
import numpy as np

# Load the trained model and preprocessing objects
model = joblib.load("trained_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
label_encoders = joblib.load("label_encoders.pkl")
selected_features = joblib.load("selected_features.pkl")


def preprocess_input(user_input):
    try:
        # Convert user input into a DataFrame
        input_df = pd.DataFrame([user_input])

        # Encode categorical features
        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))

        # Standardize and apply PCA to user input
        user_scaled = scaler.transform(input_df[selected_features])
        user_pca = pca.transform(user_scaled)

        return user_pca

    except Exception as e:
        return f"Preprocessing Error: {str(e)}"


def predict_effort(user_input):
    try:
        # Preprocess the user input
        user_pca = preprocess_input(user_input)

        # Predict effort using the trained model
        effort_prediction = model.predict(user_pca)[0]

        return round(effort_prediction, 2)

    except Exception as e:
        return f"Prediction Error: {str(e)}"


if __name__ == "__main__":
    # Example usage with user input
    user_input = {
        'Dedicated_team_members': float(input('Enter the value for Dedicated_team_members: ')),
        'Team_size': float(input('Enter the value for Team_size: ')),
        'Economic_instability_impact': float(input('Enter the value for Economic_instability_impact: ')),
        'Development_type': int(input('Enter the value for Development_type: ')),
        'Top_management_support': float(input('Enter the value for Top_management_support: ')),
        'Year_of_project': int(input('Enter the value for Year_of_project: ')),
        'Reliability_requirements': float(input('Enter the value for Reliability_requirements: ')),
        'Project_manager_experience': float(input('Enter the value for Project_manager_experience: ')),
        'User_resistance': float(input('Enter the value for User_resistance: ')),
        'User_manual': float(input('Enter the value for User_manual: ')),
        'Avg_salary': float(input('Enter the value for Avg_salary: '))
    }

    predicted_effort = predict_effort(user_input)
    print(f"\nPredicted Actual Effort based on user input: {predicted_effort:.2f}") '''

'''import joblib
import pandas as pd
import numpy as np

# Load the trained model and preprocessing objects
model = joblib.load("trained_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
label_encoders = joblib.load("label_encoders.pkl")
selected_features = joblib.load("selected_features.pkl")


def preprocess_input(user_input):
    try:
        # Convert user input into a DataFrame
        input_df = pd.DataFrame([user_input])

        # Encode categorical features
        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))

        # Standardize and apply PCA to user input
        user_scaled = scaler.transform(input_df[selected_features])
        user_pca = pca.transform(user_scaled)

        return user_pca

    except Exception as e:
        return f"Preprocessing Error: {str(e)}"


def predict_effort(user_input):
    try:
        # Preprocess the user input
        user_pca = preprocess_input(user_input)

        if isinstance(user_pca, str):
            return user_pca

        # Predict effort using the trained model
        effort_prediction = model.predict(user_pca)[0]

        return round(effort_prediction, 2)

    except Exception as e:
        return f"Prediction Error: {str(e)}"


if __name__ == "__main__":
    # Example usage with user input
    user_input = {
        'Dedicated_team_members': float(input('Enter the value for Dedicated_team_members: ')),
        'Team_size': float(input('Enter the value for Team_size: ')),
        'Economic_instability_impact': float(input('Enter the value for Economic_instability_impact: ')),
        'Development_type': int(input('Enter the value for Development_type: ')),
        'Top_management_support': float(input('Enter the value for Top_management_support: ')),
        'Year_of_project': int(input('Enter the value for Year_of_project: ')),
        'Reliability_requirements': float(input('Enter the value for Reliability_requirements: ')),
        'Project_manager_experience': float(input('Enter the value for Project_manager_experience: ')),
        'User_resistance': float(input('Enter the value for User_resistance: ')),
        'User_manual': float(input('Enter the value for User_manual: ')),
        'Avg_salary': float(input('Enter the value for Avg_salary: '))
    }

    predicted_effort = predict_effort(user_input)

    if isinstance(predicted_effort, str):
        print(f"\nError: {predicted_effort}")
    else:
        print(f"\nPredicted Actual Effort based on user input: {predicted_effort:.2f}")'''

'''import joblib
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

# Load the trained model and preprocessing objects
model = joblib.load("trained_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
label_encoders = joblib.load("label_encoders.pkl")
selected_features = joblib.load("selected_features.pkl")


def preprocess_input(user_input):
    try:
        # Convert user input into a DataFrame
        input_df = pd.DataFrame([user_input])

        # Encode categorical features, handle unseen labels
        for col, le in label_encoders.items():
            if col in input_df.columns:
                # Check if the label encoder has been fitted
                check_is_fitted(le)
                # Apply the label encoder with handling for unseen labels
                input_df[col] = input_df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1).astype(
                    int)

        # Standardize and apply PCA to user input
        user_scaled = scaler.transform(input_df[selected_features])
        user_pca = pca.transform(user_scaled)

        return user_pca

    except Exception as e:
        return f"Preprocessing Error: {str(e)}"


def predict_effort(user_input):
    try:
        # Preprocess the user input
        user_pca = preprocess_input(user_input)

        if isinstance(user_pca, str):
            return user_pca

        # Predict effort using the trained model
        effort_prediction = model.predict(user_pca)[0]

        return round(effort_prediction, 2)

    except Exception as e:
        return f"Prediction Error: {str(e)}"


if __name__ == "__main__":
    # Example usage with user input
    user_input = {
        'Dedicated_team_members': float(input('Enter the value for Dedicated_team_members: ')),
        'Team_size': float(input('Enter the value for Team_size: ')),
        'Economic_instability_impact': float(input('Enter the value for Economic_instability_impact: ')),
        'Development_type': int(input('Enter the value for Development_type: ')),
        'Top_management_support': float(input('Enter the value for Top_management_support: ')),
        'Year_of_project': int(input('Enter the value for Year_of_project: ')),
        'Reliability_requirements': float(input('Enter the value for Reliability_requirements: ')),
        'Project_manager_experience': float(input('Enter the value for Project_manager_experience: ')),
        'User_resistance': float(input('Enter the value for User_resistance: ')),
        'User_manual': float(input('Enter the value for User_manual: ')),
        'Avg_salary': float(input('Enter the value for Avg_salary: '))
    }

    predicted_effort = predict_effort(user_input)

    if isinstance(predicted_effort, str):
        print(f"\nError: {predicted_effort}")
    else:
        print(f"\nPredicted Actual Effort based on user input: {predicted_effort:.2f}")'''
'''import joblib
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

# Load trained model and preprocessing objects
model = joblib.load("trained_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
label_encoders = joblib.load("label_encoders.pkl")
selected_features = joblib.load("selected_features.pkl")


def preprocess_input(user_input):
    try:
        input_df = pd.DataFrame([user_input])

        # Encode categorical features
        for col, le in label_encoders.items():
            if col in input_df.columns:
                check_is_fitted(le)
                input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1).astype(
                    int)

        # Standardize and apply PCA
        user_scaled = scaler.transform(input_df[selected_features])
        user_pca = pca.transform(user_scaled)

        return user_pca

    except Exception as e:
        return f"Preprocessing Error: {str(e)}"


def predict_effort(user_input):
    try:
        user_pca = preprocess_input(user_input)
        if isinstance(user_pca, str):
            return user_pca
        effort_prediction = model.predict(user_pca)[0]
        return round(effort_prediction, 2)

    except Exception as e:
        return f"Prediction Error: {str(e)}"


if __name__ == "__main__":
    # Example input
    user_input = {
        'Dedicated_team_members': float(input('Enter Dedicated_team_members: ')),
        'Team_size': float(input('Enter Team_size: ')),
        'Economic_instability_impact': float(input('Enter Economic_instability_impact: ')),
        'Development_type': int(input('Enter Development_type: ')),
        'Top_management_support': float(input('Enter Top_management_support: ')),
        'Year_of_project': int(input('Enter Year_of_project: ')),
        'Reliability_requirements': float(input('Enter Reliability_requirements: ')),
        'Project_manager_experience': float(input('Enter Project_manager_experience: ')),
        'User_resistance': float(input('Enter User_resistance: ')),
        'User_manual': float(input('Enter User_manual: ')),
        'Avg_salary': float(input('Enter Avg_salary: '))
    }

    predicted_effort = predict_effort(user_input)

    if isinstance(predicted_effort, str):
        print(f"\nError: {predicted_effort}")
    else:
        print(f"\nPredicted Effort: {predicted_effort:.2f}")'''
'''import joblib
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

# Load trained model and preprocessing objects
model = joblib.load("trained_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
label_encoders = joblib.load("label_encoders.pkl")
selected_features = joblib.load("selected_features.pkl")


def preprocess_input(user_input):
    try:
        input_df = pd.DataFrame([user_input])

        # Ensure columns match training data
        input_df = input_df[selected_features]

        # Encode categorical features
        for col, le in label_encoders.items():
            if col in input_df.columns:
                check_is_fitted(le)
                input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1).astype(
                    int)

        # Standardize and apply PCA
        user_scaled = scaler.transform(input_df)
        user_pca = pca.transform(user_scaled)

        return user_pca

    except Exception as e:
        return f"Preprocessing Error: {str(e)}"


def predict_effort(user_input):
    try:
        user_pca = preprocess_input(user_input)
        if isinstance(user_pca, str):
            return user_pca
        effort_prediction = model.predict(user_pca)[0]
        return round(effort_prediction, 2)

    except Exception as e:
        return f"Prediction Error: {str(e)}"


if __name__ == "__main__":
    # Example input
    user_input = {
        'Dedicated_team_members': float(input('Enter Dedicated_team_members: ')),
        'Team_size': float(input('Enter Team_size: ')),
        'Economic_instability_impact': float(input('Enter Economic_instability_impact: ')),
        'Development_type': int(input('Enter Development_type: ')),
        'Top_management_support': float(input('Enter Top_management_support: ')),
        'Year_of_project': int(input('Enter Year_of_project: ')),
        'Reliability_requirements': float(input('Enter Reliability_requirements: ')),
        'Project_manager_experience': float(input('Enter Project_manager_experience: ')),
        'User_resistance': float(input('Enter User_resistance: ')),
        'User_manual': float(input('Enter User_manual: ')),
        'Avg_salary': float(input('Enter Avg_salary: '))
    }

    predicted_effort = predict_effort(user_input)

    if isinstance(predicted_effort, str):
        print(f"\nError: {predicted_effort}")
    else:
        print(f"\nPredicted Effort: {predicted_effort:.2f}")'''
'''import joblib
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

# Load trained model and preprocessing objects
model = joblib.load("trained_rf_model.pkl")
scaler = joblib.load("trained_rf_scaler.pkl")
pca = joblib.load("trained_rf_pca.pkl")
label_encoders = joblib.load("trained_rf_label_encoders.pkl")
selected_features = joblib.load("trained_rf_selected_features.pkl")

def preprocess_input(user_input):
    """Preprocess the user input to match the training pipeline."""
    try:
        # Convert user input to DataFrame
        input_df = pd.DataFrame([user_input])

        # Ensure columns match training data
        input_df = input_df[selected_features]

        # Encode categorical features
        for col, le in label_encoders.items():
            if col in input_df.columns:
                check_is_fitted(le)
                input_df[col] = input_df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                ).astype(int)

        # Convert to numpy array for compatibility
        input_array = input_df.values

        # Standardize input
        user_scaled = scaler.transform(input_array)
        print("\nDebug: Scaled User Input:")
        print(user_scaled)

        # Apply PCA
        user_pca = pca.transform(user_scaled)
        print("\nDebug: PCA-Transformed User Input:")
        print(user_pca)

        return user_pca

    except Exception as e:
        return f"Preprocessing Error: {str(e)}"

def predict_effort(user_input):
    """Predict the effort based on user input."""
    try:
        # Preprocess the input
        user_pca = preprocess_input(user_input)
        if isinstance(user_pca, str):  # Handle preprocessing errors
            return user_pca

        # Make the prediction
        effort_prediction = model.predict(user_pca)[0]
        return round(effort_prediction, 2)

    except Exception as e:
        return f"Prediction Error: {str(e)}"

if __name__ == "__main__":
    # Example input
    user_input = {
        'Dedicated_team_members': float(input('Enter Dedicated_team_members: ')),
        'Team_size': float(input('Enter Team_size: ')),
        'Economic_instability_impact': float(input('Enter Economic_instability_impact: ')),
        'Development_type': int(input('Enter Development_type: ')),
        'Top_management_support': float(input('Enter Top_management_support: ')),
        'Year_of_project': int(input('Enter Year_of_project: ')),
        'Reliability_requirements': float(input('Enter Reliability_requirements: ')),
        'Project_manager_experience': float(input('Enter Project_manager_experience: ')),
        'User_resistance': float(input('Enter User_resistance: ')),
        'User_manual': float(input('Enter User_manual: '))
    }

    # Predict effort
    predicted_effort = predict_effort(user_input)

    # Output the result
    if isinstance(predicted_effort, str):
        print(f"\nError: {predicted_effort}")
    else:
        print(f"\nPredicted Effort: {predicted_effort:.2f}")'''
'''import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import os

# Set a global random state for reproducibility
RANDOM_STATE = 42

# Define selected features
SELECTED_FEATURES = [
    'Dedicated_team_members', 'Team_size', 'Economic_instability_impact',
    'Development_type', 'Top_management_support', 'Year_of_project',
    'Reliability_requirements', 'Project_manager_experience', 'User_resistance',
    'User_manual'
]

def load_data(file_path):
    """Load dataset and preprocess it (handle missing values, encoding categorical data)."""
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]

    # Handle missing values
    df.replace('?', np.nan, inplace=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].astype(float).fillna(df[col].mean())  # Fill missing numeric values with mean

    # Encode categorical features
    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders

def preprocess_data(df, selected_features):
    """Select features, scale them, apply PCA, and return preprocessed data."""
    X = df[selected_features]
    y = df['Actual_effort']

    # Convert to numpy array for compatibility with scikit-learn
    X_array = X.values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_array)

    # Apply PCA
    pca = PCA(n_components=5, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, y, scaler, pca

def train_model(X, y):
    """Train a Random Forest model using the preprocessed data."""
    model = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_split=2,
        min_samples_leaf=2, max_features='sqrt', random_state=RANDOM_STATE
    )
    model.fit(X, y)
    return model

def save_model(model, scaler, pca, label_encoders, selected_features, filename_prefix="trained_rf"):
    """Save the trained model, scaler, PCA, label encoders, and selected features."""
    joblib.dump(model, f"{filename_prefix}_model.pkl")
    joblib.dump(scaler, f"{filename_prefix}_scaler.pkl")
    joblib.dump(pca, f"{filename_prefix}_pca.pkl")
    joblib.dump(label_encoders, f"{filename_prefix}_label_encoders.pkl")
    joblib.dump(selected_features, f"{filename_prefix}_selected_features.pkl")

if __name__ == "__main__":
    # Load and preprocess data
    file_path = "data/projectdata.csv"
    df, label_encoders = load_data(file_path)
    X, y, scaler, pca = preprocess_data(df, SELECTED_FEATURES)

    # Train model
    model = train_model(X, y)

    # Save model and preprocessors
    save_model(model, scaler, pca, label_encoders, SELECTED_FEATURES)

    print("Model training complete. Files saved.")'''
'''import joblib
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

# Load trained model and preprocessing objects
model = joblib.load("trained_rf_model.pkl")
scaler = joblib.load("trained_rf_scaler.pkl")
pca = joblib.load("trained_rf_pca.pkl")
label_encoders = joblib.load("trained_rf_label_encoders.pkl")
selected_features = joblib.load("trained_rf_selected_features.pkl")

def preprocess_input(user_input):
    """Preprocess the user input to match the training pipeline."""
    try:
        # Convert user input to DataFrame
        input_df = pd.DataFrame([user_input])

        # Ensure columns match training data
        input_df = input_df[selected_features]

        # Encode categorical features
        for col, le in label_encoders.items():
            if col in input_df.columns:
                check_is_fitted(le)
                input_df[col] = input_df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                ).astype(int)

        # Convert to numpy array for compatibility
        input_array = input_df.values

        # Standardize input
        user_scaled = scaler.transform(input_array)
        print("\nDebug: Scaled User Input:")
        print(user_scaled)

        # Apply PCA
        user_pca = pca.transform(user_scaled)
        print("\nDebug: PCA-Transformed User Input:")
        print(user_pca)

        return user_pca

    except Exception as e:
        return f"Preprocessing Error: {str(e)}"

def predict_effort(user_input):
    """Predict the effort based on user input."""
    try:
        # Preprocess the input
        user_pca = preprocess_input(user_input)
        if isinstance(user_pca, str):  # Handle preprocessing errors
            return user_pca

        # Make the prediction
        effort_prediction = model.predict(user_pca)[0]
        return round(effort_prediction, 2)

    except Exception as e:
        return f"Prediction Error: {str(e)}"

if __name__ == "__main__":
    # Example input
    user_input = {
        'Dedicated_team_members': float(input('Enter Dedicated_team_members: ')),
        'Team_size': float(input('Enter Team_size: ')),
        'Economic_instability_impact': float(input('Enter Economic_instability_impact: ')),
        'Development_type': int(input('Enter Development_type: ')),
        'Top_management_support': float(input('Enter Top_management_support: ')),
        'Year_of_project': int(input('Enter Year_of_project: ')),
        'Reliability_requirements': float(input('Enter Reliability_requirements: ')),
        'Project_manager_experience': float(input('Enter Project_manager_experience: ')),
        'User_resistance': float(input('Enter User_resistance: ')),
        'User_manual': float(input('Enter User_manual: '))
    }

    # Predict effort
    predicted_effort = predict_effort(user_input)

    # Output the result
    if isinstance(predicted_effort, str):
        print(f"\nError: {predicted_effort}")
    else:
        print(f"\nPredicted Effort: {predicted_effort:.2f}")'''
'''import joblib
import pandas as pd

# Load trained model and preprocessing objects
model = joblib.load("trained_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
label_encoders = joblib.load("label_encoders.pkl")
selected_features = joblib.load("selected_features.pkl")

def preprocess_user_input(user_input, selected_features, scaler, pca):
    """Preprocess user input to match the training pipeline."""
    input_df = pd.DataFrame([user_input])
    input_df = input_df[selected_features]
    input_scaled = scaler.transform(input_df)
    input_pca = pca.transform(input_scaled)
    return input_pca

# Example usage for prediction
if __name__ == "__main__":
    user_input = {
        'Dedicated_team_members': 6,
        'Team_size': 6,
        'Economic_instability_impact': 1,
        'Development_type': 1,
        'Top_management_support': 4,
        'Year_of_project': 2015,
        'Reliability_requirements': 3,
        'Project_manager_experience': 2,
        'User_resistance': 2,
        'User_manual': 1
    }

    # Preprocess user input
    user_pca = preprocess_user_input(user_input, selected_features, scaler, pca)

    # Predict effort
    prediction = model.predict(user_pca)
    print(f"Predicted Effort: {prediction[0]:.2f}")'''

import pandas as pd
import numpy as np
import pickle
import os

# Load saved model, scaler, and selected features
with open("random_forest_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
with open("selected_features.pkl", "rb") as features_file:
    selected_features = pickle.load(features_file)


# Function to predict effort
def predict_effort(user_input):
    # Convert user input to DataFrame
    user_df = pd.DataFrame([user_input])

    # Ensure input has the correct order of selected features
    user_df = user_df[selected_features]

    # Scale input using the saved scaler
    user_scaled = scaler.transform(user_df)

    # Predict effort
    predicted_effort = model.predict(user_scaled)
    return predicted_effort[0]


# Get user input
print("Enter values for the following features:")
user_input = {}
for feature in selected_features:
    while True:
        try:
            user_input[feature] = float(input(f"{feature}: "))
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Predict and display the result
predicted_effort = predict_effort(user_input)
print(f"\nPredicted Actual Effort: {predicted_effort}")
