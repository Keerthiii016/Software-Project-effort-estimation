'''import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import os

# Set a global random state for reproducibility
RANDOM_STATE = 42

# Define selected features (Ensure these match the ones used in training)
SELECTED_FEATURES = [
'Dedicated_team_members', 'Team_size', 'Economic_instability_impact',
                     'Development_type', 'Top_management_support', 'Year_of_project',
                     'Reliability_requirements', 'Project_manager_experience', 'User_resistance',
                     'User_manual'
]


def load_data(file_path):
    """
    Load dataset and preprocess it (handle missing values, encoding categorical data).
    """
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
    """ Select features, scale them, apply PCA, and return preprocessed data. """
    X = df[selected_features]
    y = df['Actual_effort']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=5, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    return X_pca, y, scaler, pca


def train_model(X, y):
    """ Train a Random Forest model using the preprocessed data. """
    model = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_split=2,
        min_samples_leaf=2, max_features='sqrt', random_state=RANDOM_STATE
    )
    model.fit(X, y)
    return model


def save_model(model, scaler, pca, label_encoders, selected_features, filename_prefix="trained_rf"):
    """ Save the trained model, scaler, PCA, label encoders, and selected features. """
    joblib.dump(model, f"{filename_prefix}_model.pkl")
    joblib.dump(scaler, f"{filename_prefix}_scaler.pkl")
    joblib.dump(pca, f"{filename_prefix}_pca.pkl")
    joblib.dump(label_encoders, f"{filename_prefix}_label_encoders.pkl")
    joblib.dump(selected_features, "selected_features.pkl")  # Saving selected features


def load_model(filename_prefix="trained_rf"):
    """ Load the trained model, scaler, PCA, label encoders, and selected features. """
    model = joblib.load(f"{filename_prefix}_model.pkl")
    scaler = joblib.load(f"{filename_prefix}_scaler.pkl")
    pca = joblib.load(f"{filename_prefix}_pca.pkl")
    label_encoders = joblib.load(f"{filename_prefix}_label_encoders.pkl")
    selected_features = joblib.load("selected_features.pkl")

    return model, scaler, pca, label_encoders, selected_features


if __name__ == "__main__":
    # Load and preprocess data
    file_path = os.path.expanduser("data/projectdata.csv")
    # Update file path if needed
    df, label_encoders = load_data(file_path)
    X, y, scaler, pca = preprocess_data(df, SELECTED_FEATURES)

    # Train model
    model = train_model(X, y)

    # Save model, preprocessors, and selected features
    save_model(model, scaler, pca, label_encoders, SELECTED_FEATURES)

    print("Model training complete. Files saved.")'''
'''import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import os

# Set a global random state for reproducibility
RANDOM_STATE = 42

# Define selected features (Ensure these match the ones used in training)
SELECTED_FEATURES = [
    'Dedicated_team_members', 'Team_size', 'Economic_instability_impact',
    'Development_type', 'Top_management_support', 'Year_of_project',
    'Reliability_requirements', 'Project_manager_experience', 'User_resistance',
    'User_manual'
]


def load_data(file_path):
    """
    Load dataset and preprocess it (handle missing values, encoding categorical data).
    """
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
    """ Select features, scale them, apply PCA, and return preprocessed data. """
    X = df[selected_features]
    y = df['Actual_effort']

    # Scale features
    scaler = StandardScaler()
    print("Training Features Before Scaling:", X.head())
    X_scaled = scaler.fit_transform(X)
    print("Training Features After Scaling:", X_scaled[:5])

    # Apply PCA
    pca = PCA(n_components=5, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    print("Training Features After PCA:", X_pca[:5])

    return X_pca, y, scaler, pca


def train_model(X, y):
    """ Train a Random Forest model using the preprocessed data. """
    model = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_split=2,
        min_samples_leaf=2, max_features='sqrt', random_state=RANDOM_STATE
    )
    model.fit(X, y)
    return model


def save_model(model, scaler, pca, label_encoders, selected_features, filename_prefix="trained_rf"):
    """ Save the trained model, scaler, PCA, label encoders, and selected features. """
    joblib.dump(model, f"{filename_prefix}_model.pkl")
    joblib.dump(scaler, f"{filename_prefix}_scaler.pkl")
    joblib.dump(pca, f"{filename_prefix}_pca.pkl")
    joblib.dump(label_encoders, f"{filename_prefix}_label_encoders.pkl")
    joblib.dump(selected_features, "selected_features.pkl")  # Saving selected features


def load_model(filename_prefix="trained_rf"):
    """ Load the trained model, scaler, PCA, label encoders, and selected features. """
    model = joblib.load(f"{filename_prefix}_model.pkl")
    scaler = joblib.load(f"{filename_prefix}_scaler.pkl")
    pca = joblib.load(f"{filename_prefix}_pca.pkl")
    label_encoders = joblib.load(f"{filename_prefix}_label_encoders.pkl")
    selected_features = joblib.load("selected_features.pkl")

    return model, scaler, pca, label_encoders, selected_features


if __name__ == "__main__":
    # Load and preprocess data
    file_path = os.path.expanduser("data/projectdata.csv")  # Update file path if needed
    df, label_encoders = load_data(file_path)
    X, y, scaler, pca = preprocess_data(df, SELECTED_FEATURES)

    # Train model
    model = train_model(X, y)

    # Save model, preprocessors, and selected features
    save_model(model, scaler, pca, label_encoders, SELECTED_FEATURES)

    print("Model training complete. Files saved.")'''
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
    """ Load dataset and preprocess it (handle missing values, encoding categorical data). """
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
    """ Select features, scale them, apply PCA, and return preprocessed data. """
    X = df[selected_features]
    y = df['Actual_effort']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=5, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, y, scaler, pca

def train_model(X, y):
    """ Train a Random Forest model using the preprocessed data. """
    model = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_split=2,
        min_samples_leaf=2, max_features='sqrt', random_state=RANDOM_STATE
    )
    model.fit(X, y)
    return model

def save_model(model, scaler, pca, label_encoders, selected_features, filename_prefix="trained_rf"):
    """ Save the trained model, scaler, PCA, label encoders, and selected features. """
    joblib.dump(model, f"{filename_prefix}_model.pkl")
    joblib.dump(scaler, f"{filename_prefix}_scaler.pkl")
    joblib.dump(pca, f"{filename_prefix}_pca.pkl")
    joblib.dump(label_encoders, f"{filename_prefix}_label_encoders.pkl")
    joblib.dump(selected_features, "selected_features.pkl")

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
    X_array = X.values  # Use .values to remove feature names

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
'''import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import optuna
from optuna.samplers import TPESampler

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load dataset
file_path = os.path.expanduser("data/projectdata.csv")
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip().str.replace(" ", "_")

df.replace('?', np.nan, inplace=True)
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].astype(float)
    df[col].fillna(df[col].mean(), inplace=True)

# Encode categorical features
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['Actual_effort', 'Estimated_size', 'Degree_of_standards_usage'])
y = df['Actual_effort']

# Feature selection using RandomForest
rf_selector = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, bootstrap=False)
rf_selector.fit(X, y)
feature_importances = rf_selector.feature_importances_
indices = np.argsort(feature_importances)[::-1][:10]
selected_features = X.columns[indices].tolist()

X_selected = X[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Apply PCA
pca = PCA(n_components=5, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=RANDOM_STATE)

# Hyperparameter tuning

def objective(trial):
    model = RandomForestRegressor(
        n_estimators=trial.suggest_int('n_estimators', 100, 500),
        max_depth=trial.suggest_int('max_depth', 10, 30),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 4),
        max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        random_state=RANDOM_STATE,
        bootstrap=False
    )
    model.fit(X_train, y_train)
    return np.mean(abs(y_test - model.predict(X_test)))

study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=50)

best_rf_model = RandomForestRegressor(**study.best_params, random_state=RANDOM_STATE, bootstrap=False)
best_rf_model.fit(X_train, y_train)

# Save components
joblib.dump(best_rf_model, "trained_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(selected_features, "selected_features.pkl")

print("Model and preprocessing objects saved successfully.")'''
'''#import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from optuna import create_study
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load dataset
file_path = os.path.expanduser("data/projectdata.csv")
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Handle missing values
df.replace("?", np.nan, inplace=True)
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].astype(float)
    df[col].fillna(df[col].mean(), inplace=True)

# Label Encoding for categorical features
label_encoders = {}
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=["Actual_effort", "Estimated_size", "Degree_of_standards_usage"])
y = df["Actual_effort"]

# Feature selection
rf_selector = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, bootstrap=False)
rf_selector.fit(X, y)
feature_importances = rf_selector.feature_importances_
indices = np.argsort(feature_importances)[::-1][:10]
selected_feature_names = X.columns[indices].tolist()

# Save selected features for later
joblib.dump(selected_feature_names, "selected_features.pkl")

# Standardize selected features
X_selected = X[selected_feature_names]
scaler = StandardScaler()
scaler.fit(X_selected)
X_scaled = scaler.transform(X_selected)
joblib.dump(scaler, "scaler.pkl")

# Apply PCA for dimensionality reduction
pca = PCA(n_components=5, random_state=RANDOM_STATE)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
joblib.dump(pca, "pca.pkl")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
)

# Hyperparameter tuning with Optuna
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    max_depth = trial.suggest_int("max_depth", 10, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=RANDOM_STATE,
        bootstrap=False
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

study = create_study(direction="minimize", sampler=TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=50)
best_params = study.best_params

# Train the best model
best_model = RandomForestRegressor(**best_params, random_state=RANDOM_STATE, bootstrap=False)
best_model.fit(X_train, y_train)
joblib.dump(best_model, "trained_rf_model.pkl")

print(f"Model, scaler, PCA, and selected features saved successfully!")#'''
'''import pandas as pd
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import optuna
from optuna.samplers import TPESampler

# Set a global random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load dataset
file_path = os.path.expanduser("~/Downloads/projectdata.csv")  # Update path if needed
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Handle missing values
df.replace('?', np.nan, inplace=True)
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].astype(float)
    df[col].fillna(df[col].mean(), inplace=True)

# Label Encoding for categorical features
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['Actual_effort', 'Estimated_size', 'Degree_of_standards_usage'])
y = df['Actual_effort']

# Data Augmentation using Gaussian noise
np.random.seed(RANDOM_STATE)
noise_factor = 0.05
X_augmented = X + noise_factor * np.random.normal(size=X.shape)
y_augmented = y + noise_factor * np.random.normal(size=y.shape)

# Combine original and augmented data
X_combined = np.vstack((X, X_augmented))
y_combined = np.hstack((y, y_augmented))

# Feature selection using RandomForestRegressor
rf_selector = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, bootstrap=False)
rf_selector.fit(X_combined, y_combined)

# Get feature importances and select top 10 features
feature_importances = rf_selector.feature_importances_
indices = np.argsort(feature_importances)[::-1][:10]
selected_feature_names = X.columns[indices].tolist()

print("Selected Features for Prediction:", selected_feature_names)

# Select the top 10 features
X_selected = X_combined[:, indices]

# Standardize features while maintaining feature names
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pd.DataFrame(X_selected, columns=[X.columns[i] for i in indices]))

# Apply PCA for dimensionality reduction
pca = PCA(n_components=5, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_combined, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
)

# Hyperparameter tuning with Optuna
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 10, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=RANDOM_STATE,
        bootstrap=False
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

# Create a study and optimize
study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=50)
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Train best model with tuned parameters
best_rf_model = RandomForestRegressor(
    **best_params, random_state=RANDOM_STATE, bootstrap=False
)
best_rf_model.fit(X_train, y_train)

# Evaluate the tuned model on the test set
y_pred_rf = best_rf_model.predict(X_test)
mae_tuned = mean_absolute_error(y_test, y_pred_rf)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_tuned = r2_score(y_test, y_pred_rf)

print(f"Tuned MAE: {mae_tuned:.4f}")
print(f"Tuned RMSE: {rmse_tuned:.4f}")
print(f"Tuned R² Score: {r2_tuned:.4f}")

# Predict on the original dataset
X_original = df[selected_feature_names]
X_original_scaled = scaler.transform(pd.DataFrame(X_original, columns=selected_feature_names))
X_original_pca = pca.transform(X_original_scaled)

y_original_pred = best_rf_model.predict(X_original_pca)
results_df = pd.DataFrame({
    'Actual_Effort': y,
    'Predicted_Effort': y_original_pred
})

print("\nPredictions for the Original Dataset:")
print(results_df.head(10))

# Take user input for prediction
user_input = {}
for feature in selected_feature_names:
    user_input[feature] = float(input(f"Enter the value for {feature}: "))

user_df = pd.DataFrame([user_input])
user_scaled = scaler.transform(pd.DataFrame(user_df, columns=selected_feature_names))
user_pca = pca.transform(user_scaled)

print("\nDebug - Transformed User Input (PCA):", user_pca)

# Predict using the tuned model
user_pred = best_rf_model.predict(user_pca)
print(f"\nPredicted Actual Effort based on user input: {user_pred[0]:.4f}")

# Save models and preprocessing tools
joblib.dump(best_rf_model, "trained_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(selected_feature_names, "selected_features.pkl")'''

'''import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBRegressor

# Fix random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load dataset
file_path = os.path.expanduser("~/Downloads/projectdata.csv")  # Update the path if needed
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Handle missing values
df.replace('?', np.nan, inplace=True)
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].astype(float)
    df[col] = df[col].fillna(df[col].mean())

# Label Encoding for categorical features
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['Actual_effort', 'Estimated_size', 'Degree_of_standards_usage'])
y = df['Actual_effort']

# Data Augmentation: Add Gaussian noise
noise_factor = 0.1  # Adjust noise factor as needed
X_augmented = X + noise_factor * np.random.normal(size=X.shape)
y_augmented = y + noise_factor * np.random.normal(size=y.shape)

# Combine original and augmented data
X_combined = np.vstack((X, X_augmented))
y_combined = np.hstack((y, y_augmented))

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Feature selection using RandomForestRegressor
rf_selector = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
rf_selector.fit(X_scaled, y_combined)
feature_importances = rf_selector.feature_importances_
indices = np.argsort(feature_importances)[::-1][:10]  # Get indices of top 10 features

# Select the top 10 features
X_selected = X_scaled[:, indices]
selected_feature_names = X.columns[indices]

print("Selected Features for Prediction:", list(selected_feature_names))

# Apply PCA to the selected features
n_components = min(X_selected.shape[0], X_selected.shape[1], 5)  # Dynamically adjust components
pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X_selected)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_combined, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)

# Hyperparameter tuning with Optuna for XGBRegressor
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae

# Create a study and optimize
study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=100)
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Train the best XGBRegressor model
best_xgb_model = XGBRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    random_state=RANDOM_SEED
)
best_xgb_model.fit(X_train, y_train)

# Evaluate the tuned model on the test set
y_pred_xgb = best_xgb_model.predict(X_test)
mae_tuned = mean_absolute_error(y_test, y_pred_xgb)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_tuned = r2_score(y_test, y_pred_xgb)

print(f"Tuned MAE: {mae_tuned}")
print(f"Tuned RMSE: {rmse_tuned}")
print(f"Tuned R² Score: {r2_tuned}")

# Save the trained model, scaler, PCA, and selected features
joblib.dump(best_xgb_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')
joblib.dump(list(selected_feature_names), 'selected_features.pkl')'''
'''import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBRegressor

# Fix random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load dataset
file_path = os.path.expanduser("~/Downloads/projectdata.csv")  # Update the path if needed
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Handle missing values
df.replace('?', np.nan, inplace=True)
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].astype(float)
    df[col] = df[col].fillna(df[col].mean())

# Label Encoding for categorical features
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['Actual_effort', 'Estimated_size', 'Degree_of_standards_usage'])
y = df['Actual_effort']

# Data Augmentation: Add Gaussian noise
noise_factor = 0.1  # Adjust noise factor as needed
X_augmented = X + noise_factor * np.random.normal(size=X.shape)
y_augmented = y + noise_factor * np.random.normal(size=y.shape)

# Combine original and augmented data
X_combined = np.vstack((X, X_augmented))
y_combined = np.hstack((y, y_augmented))

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Feature selection using RandomForestRegressor
rf_selector = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
rf_selector.fit(X_scaled, y_combined)
feature_importances = rf_selector.feature_importances_
indices = np.argsort(feature_importances)[::-1][:10]  # Get indices of top 10 features

# Select the top 10 features
X_selected = X_scaled[:, indices]
selected_feature_names = X.columns[indices]

print("Selected Features for Prediction:", list(selected_feature_names))

# Fit the StandardScaler only for the selected features
scaler_selected = StandardScaler()
X_selected_scaled = scaler_selected.fit_transform(X_selected)

# Apply PCA to the selected features (Dynamic adjustment of n_components)
n_components = min(X_selected_scaled.shape[0], X_selected_scaled.shape[1], 5)
pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X_selected_scaled)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_combined, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)

# Hyperparameter tuning with Optuna for XGBRegressor
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae

# Create a study and optimize
study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=100)
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Train the best XGBRegressor model
best_xgb_model = XGBRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    random_state=RANDOM_SEED
)
best_xgb_model.fit(X_train, y_train)

# Evaluate the tuned model on the test set
y_pred_xgb = best_xgb_model.predict(X_test)
mae_tuned = mean_absolute_error(y_test, y_pred_xgb)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_tuned = r2_score(y_test, y_pred_xgb)

print(f"Tuned MAE: {mae_tuned}")
print(f"Tuned RMSE: {rmse_tuned}")
print(f"Tuned R² Score: {r2_tuned}")

# Save the trained model, scaler, PCA, and selected features
joblib.dump(best_xgb_model, 'model.pkl')  # Save the model
joblib.dump(scaler_selected, 'scaler.pkl')  # Save the scaler fitted for selected features
joblib.dump(pca, 'pca.pkl')  # Save the PCA transformer
joblib.dump(list(selected_feature_names), 'selected_features.pkl')  # Save selected feature names

print("✅ Model, scaler, PCA, and selected features saved successfully!")
'''
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna

# Fix random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load dataset
file_path = os.path.expanduser("~/Downloads/projectdata.csv")  # Update the path if needed
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Drop unnecessary columns
df.drop(columns=['Estimated_size', 'Degree_of_standards_usage'], inplace=True)

# Handle missing values
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Separate numeric and categorical columns
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=['object']).columns

df[num_cols] = num_imputer.fit_transform(df[num_cols])
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# Label Encoding for categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['Actual_effort'])
y = df['Actual_effort']

# Data Augmentation: Add Gaussian noise
noise_factor = 0.1
X_augmented = X + noise_factor * np.random.normal(size=X.shape)
y_augmented = y + noise_factor * np.random.normal(size=y.shape)

# Combine original and augmented data
X_combined = np.vstack((X, X_augmented))
y_combined = np.hstack((y, y_augmented))

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Feature Selection using RFE with RandomForest
rf_model = RandomForestRegressor(random_state=RANDOM_SEED)
rfe = RFE(estimator=rf_model, n_features_to_select=10)
X_rfe = rfe.fit_transform(X_scaled, y_combined)

selected_features = X.columns[rfe.get_support()]
print(f"Selected Features: {list(selected_features)}")

# Save selected features
with open("selected_features.pkl", "wb") as f:
    pickle.dump(list(selected_features), f)

# Reinitialize Scaler for Selected Features
scaler_selected = StandardScaler()
X_rfe_scaled = scaler_selected.fit_transform(X_rfe)

# Hyperparameter tuning for Random Forest using Optuna
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)

    rf_model_tuned = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_SEED
    )
    rf_model_tuned.fit(X_rfe_scaled, y_combined)
    scores = cross_val_score(rf_model_tuned, X_rfe_scaled, y_combined, cv=5, scoring='neg_mean_squared_error')
    return -1 * scores.mean()

study = optuna.create_study(direction='minimize', study_name='Random Forest Tuning')
study.optimize(objective, n_trials=50)
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Train final Random Forest model with best hyperparameters
final_rf_model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=RANDOM_SEED
)
final_rf_model.fit(X_rfe_scaled, y_combined)

# Save the trained model and scaler
with open("random_forest_model.pkl", "wb") as model_file:
    pickle.dump(final_rf_model, model_file)
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler_selected, scaler_file)

print("Model and scaler saved successfully.")
