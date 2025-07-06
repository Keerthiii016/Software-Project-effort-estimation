'''import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import optuna
from optuna.samplers import TPESampler
import joblib

# Set a global random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load dataset
file_path = os.path.expanduser("data/projectdata.csv")
  # Update path if needed
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

# Data Augmentation using Gaussian noise (with fixed seed)
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

# Select the top 10 features before scaling
X_selected = X_combined[:, indices]

# Standardize features only for the selected features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Apply PCA for dimensionality reduction (Fixed components)
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
        bootstrap=False  # Disable bootstrap to reduce variance
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

# Create a study and optimize
study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=50)  # Reduced trials for faster execution
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
X_original = df[selected_feature_names]  # Select only the chosen features
X_original_scaled = scaler.transform(X_original)  # Scale only selected features
X_original_pca = pca.transform(X_original_scaled)  # Apply PCA

# Predict using the tuned model
y_original_pred = best_rf_model.predict(X_original_pca)
results_df = pd.DataFrame({
    'Actual_Effort': y,
    'Predicted_Effort': y_original_pred
})

# Save the trained model, scaler, PCA, and label encoders
joblib.dump(best_rf_model, "trained_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(selected_feature_names, "selected_features.pkl")


# Print the results
print("\nPredictions for the Original Dataset:")
print(results_df.head(10))  # Print the first 10 rows for comparison'''

'''import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import optuna
from optuna.samplers import TPESampler
import joblib

# Set a global random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load dataset
file_path = os.path.expanduser("data/projectdata.csv")  # Update path if needed
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Handle missing values
df.replace('?', np.nan, inplace=True)
df = df.astype(float, errors='ignore')  # Convert numeric columns

for col in df.select_dtypes(include=[np.number]).columns:
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

# Data Augmentation using Gaussian noise (with fixed seed)
noise_factor = 0.05
X_augmented = X + noise_factor * np.random.normal(size=X.shape)
y_augmented = y + noise_factor * np.random.normal(size=y.shape)

# Combine original and augmented data
X_combined = pd.concat([X, pd.DataFrame(X_augmented, columns=X.columns)], axis=0).reset_index(drop=True)
y_combined = np.hstack((y, y_augmented))

# Feature selection using RandomForestRegressor
rf_selector = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, bootstrap=False)
rf_selector.fit(X_combined, y_combined)

# Get feature importances and select top 10 features
feature_importances = rf_selector.feature_importances_
indices = np.argsort(feature_importances)[::-1][:10]
selected_feature_names = X.columns[indices].tolist()

print("Selected Features for Prediction:", selected_feature_names)

# Select the top 10 features before scaling
X_selected = X_combined[selected_feature_names]

# Standardize features only for the selected features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Apply PCA for dimensionality reduction (Fixed components)
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

# Optimize using Optuna
study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=50)  # Reduced trials for faster execution
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Train best model
best_rf_model = RandomForestRegressor(**best_params, random_state=RANDOM_STATE, bootstrap=False)
best_rf_model.fit(X_train, y_train)

# Save objects
joblib.dump(best_rf_model, "trained_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(selected_feature_names, "selected_features.pkl")'''
'''import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import optuna
from optuna.samplers import TPESampler
import joblib

# Set a global random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load dataset
file_path = os.path.expanduser("data/projectdata.csv")  # Update path if needed
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Handle missing values
df.replace('?', np.nan, inplace=True)
df = df.astype(float, errors='ignore')  # Convert numeric columns

for col in df.select_dtypes(include=[np.number]).columns:
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

# Feature selection using RandomForestRegressor
rf_selector = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, bootstrap=False)
rf_selector.fit(X, y)

# Get feature importances and select top 10 features
feature_importances = rf_selector.feature_importances_
indices = np.argsort(feature_importances)[::-1][:10]
selected_feature_names = X.columns[indices].tolist()

print("Selected Features for Prediction:", selected_feature_names)

# Select the top 10 features before scaling
X_selected = X[selected_feature_names]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=5, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
)

# Train best model
best_rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, bootstrap=False)
best_rf_model.fit(X_train, y_train)
y_pred_rf = best_rf_model.predict(X_test)
mae_tuned = mean_absolute_error(y_test, y_pred_rf)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_tuned = r2_score(y_test, y_pred_rf)
print(f"Tuned MAE: {mae_tuned:.4f}")
print(f"Tuned RMSE: {rmse_tuned:.4f}")
print(f"Tuned R² Score: {r2_tuned:.4f}")

# Save objects
joblib.dump(best_rf_model, "trained_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(selected_feature_names, "selected_features.pkl")
'''
'''import joblib
import pandas as pd
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
file_path = os.path.expanduser("data/projectdata.csv")
#file_path = os.path.expanduser("~/Downloads/projectdata.csv")  # Update path if needed
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
X = df.drop(columns=['Actual_effort', 'Estimated_size', 'Degree_of_standards_usage',])
y = df['Actual_effort']

# Data Augmentation using Gaussian noise (with fixed seed)
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

# Select the top 10 features **before scaling**
X_selected = X_combined[:, indices]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Apply PCA for dimensionality reduction (Fixed components)
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
        bootstrap=False  # Disable bootstrap to reduce variance
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)
study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=50)  # Reduced trials for faster execution
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
X_original = df[selected_feature_names]  # Select only the chosen features
X_original_scaled = scaler.transform(X_original)  # Scale only selected features
X_original_pca = pca.transform(X_original_scaled)  # Apply PCA

# Predict using the tuned model
y_original_pred = best_rf_model.predict(X_original_pca)
results_df = pd.DataFrame({
    'Actual_Effort': y,
    'Predicted_Effort': y_original_pred
})

print("\nPredictions for the Original Dataset:")
print(results_df.head(10))  # Print the first 10 rows for comparison

# Take user input for prediction
user_input = {}
for feature in selected_feature_names:
    user_input[feature] = float(input(f"Enter the value for {feature}: "))

# Create a DataFrame for the user input
user_df = pd.DataFrame([user_input])

# Standardize and transform user input using selected features
user_scaled = scaler.transform(user_df)
user_pca = pca.transform(user_scaled)

# Predict using the tuned model
user_pred = best_rf_model.predict(user_pca)
print(f"\nPredicted Actual Effort based on user input: {user_pred[0]:.4f}")
joblib.dump(best_rf_model, "trained_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(SELECTED_FEATURES, "selected_features.pkl")'''

'''import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import optuna
from optuna.samplers import TPESampler
import os

# Set a global random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Define selected features
SELECTED_FEATURES = [
    'Dedicated_team_members', 'Team_size', 'Economic_instability_impact',
    'Development_type', 'Top_management_support', 'Year_of_project',
    'Reliability_requirements', 'Project_manager_experience', 'User_resistance',
    'User_manual'
]

# Load and preprocess the dataset
file_path = os.path.expanduser("data/projectdata.csv")
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Handle missing values
df.replace('?', np.nan, inplace=True)
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].astype(float).fillna(df[col].mean(), inplace=True)

# Label Encoding for categorical features
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define features and target
X = df[SELECTED_FEATURES]
y = df['Actual_effort']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=5, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
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
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=50)
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Train the best model
best_rf_model = RandomForestRegressor(**best_params, random_state=RANDOM_STATE)
best_rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred_rf = best_rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_rf)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2 = r2_score(y_test, y_pred_rf)

print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Save the model and preprocessing objects
joblib.dump(best_rf_model, "trained_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(SELECTED_FEATURES, "selected_features.pkl")'''
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

# Data Augmentation using Gaussian noise (with fixed seed)
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

# Select the top 10 features *before scaling*
X_selected = X_combined[:, indices]

# Standardize features while maintaining feature names
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pd.DataFrame(X_selected, columns=[X.columns[i] for i in indices]))

# Apply PCA for dimensionality reduction (Fixed components)
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
        bootstrap=False  # Disable bootstrap to reduce variance
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

# Create a study and optimize
study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=50)  # Reduced trials for faster execution
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
X_original = df[selected_feature_names]  # Select only the chosen features
X_original_scaled = scaler.transform(pd.DataFrame(X_original, columns=selected_feature_names))
X_original_pca = pca.transform(X_original_scaled)  # Apply PCA

# Predict using the tuned model
y_original_pred = best_rf_model.predict(X_original_pca)
results_df = pd.DataFrame({
    'Actual_Effort': y,
    'Predicted_Effort': y_original_pred
})

# Print the results
print("\nPredictions for the Original Dataset:")
print(results_df.head(10))  # Print the first 10 rows for comparison

# Take user input for prediction
user_input = {}
for feature in selected_feature_names:
    user_input[feature] = float(input(f"Enter the value for {feature}: "))

# Create a DataFrame for the user input
user_df = pd.DataFrame([user_input])

# Standardize and transform user input using selected features
user_scaled = scaler.transform(pd.DataFrame(user_df, columns=selected_feature_names))
user_pca = pca.transform(user_scaled)

# Predict using the tuned model
user_pred = best_rf_model.predict(user_pca)
print(f"\nPredicted Actual Effort based on user input: {user_pred[0]:.4f}")

# Save models and parameters
joblib.dump(best_rf_model, "trained_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(selected_feature_names, "selected_features.pkl")  # Corrected variable'''

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
file_path = os.path.expanduser("data/projectdata.csv")
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

# Select the top 10 features *before scaling*
X_selected = X_combined[:, indices]
scaler = StandardScaler()
scaler.fit(X_selected)  # Fit scaler on selected features
X_scaled = scaler.transform(X_selected)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=5, random_state=RANDOM_STATE)
pca.fit(X_scaled)  # Fit PCA on scaled features
X_pca = pca.transform(X_scaled)

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

study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=50)
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Train the best model
best_rf_model = RandomForestRegressor(**best_params, random_state=RANDOM_STATE, bootstrap=False)
best_rf_model.fit(X_train, y_train)

# Evaluate the tuned model
y_pred_rf = best_rf_model.predict(X_test)
mae_tuned = mean_absolute_error(y_test, y_pred_rf)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_tuned = r2_score(y_test, y_pred_rf)

print(f"Tuned MAE: {mae_tuned:.4f}")
print(f"Tuned RMSE: {rmse_tuned:.4f}")
print(f"Tuned R² Score: {r2_tuned:.4f}")

# Predict on the original dataset
X_original = df[selected_feature_names].values  # Convert to NumPy array
X_original_scaled = scaler.transform(X_original)
X_original_pca = pca.transform(X_original_scaled)

# Predict using the tuned model
y_original_pred = best_rf_model.predict(X_original_pca)
results_df = pd.DataFrame({
    'Actual_Effort': y,
    'Predicted_Effort': y_original_pred
})

print("\nPredictions for the Original Dataset:")
print(results_df.head(10))



# Fetch the selected row from the original dataset



# Predict using the tuned model

# Take user input for prediction
user_input = {feature: float(input(f"Enter the value for {feature}: ")) for feature in selected_feature_names}
user_df = pd.DataFrame([user_input])
user_values = user_df.values  # Convert to NumPy array

# Debug user input
print("\nDebug: User input DataFrame (before scaling):")
print(user_df)

# Standardize user input
user_scaled = scaler.transform(user_values)
print("\nDebug: Scaled user input:")
print(user_scaled)

# Apply PCA to user input
user_pca = pca.transform(user_scaled)
print("\nDebug: PCA-transformed user input:")
print(user_pca)

# Predict using the tuned model
user_pred = best_rf_model.predict(user_pca)
print(f"\nPredicted Actual Effort based on user input: {user_pred[0]:.4f}")

# Compare user input with the selected row
if np.allclose(user_values, original_row):
    print("\n✅ User input matches the original dataset row.")
else:
    print("\n❌ User input does NOT match the original dataset row.")

# Save the model, scaler, PCA, and label encoders
joblib.dump(best_rf_model, "trained_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

# Ensure feature names are saved exactly as used in training
selected_features = X.columns[indices].tolist()
joblib.dump(selected_features, "selected_features.pkl")

print("✅ Model, Scaler, PCA, and Selected Features saved successfully!")'''
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
file_path = os.path.expanduser("data/projectdata.csv")
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

# Feature selection using RandomForestRegressor
rf_selector = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, bootstrap=False)
rf_selector.fit(X, y)

# Get feature importances and select top 10 features
feature_importances = rf_selector.feature_importances_
indices = np.argsort(feature_importances)[::-1][:10]
selected_feature_names = X.columns[indices].tolist()

print("Selected Features for Prediction:", selected_feature_names)

# Select the top 10 features before scaling
X_selected = X[selected_feature_names]

# Fit scaler and PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

pca = PCA(n_components=5, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

# Data Augmentation using Gaussian noise
noise_factor = 0.05
X_augmented = X_selected + noise_factor * np.random.normal(size=X_selected.shape)
y_augmented = y + noise_factor * np.random.normal(size=y.shape)

# Combine original and augmented data
X_combined = np.vstack((X_selected, X_augmented))
y_combined = np.hstack((y, y_augmented))

# Scale and transform the combined data
X_combined_scaled = scaler.transform(X_combined)
X_combined_pca = pca.transform(X_combined_scaled)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_combined_pca, y_combined, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
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

study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=50)
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Train the best model
best_rf_model = RandomForestRegressor(**best_params, random_state=RANDOM_STATE, bootstrap=False)
best_rf_model.fit(X_train, y_train)

# Evaluate the tuned model
y_pred_rf = best_rf_model.predict(X_test)
mae_tuned = mean_absolute_error(y_test, y_pred_rf)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_tuned = r2_score(y_test, y_pred_rf)

print(f"Tuned MAE: {mae_tuned:.4f}")
print(f"Tuned RMSE: {rmse_tuned:.4f}")
print(f"Tuned R² Score: {r2_tuned:.4f}")

# Save the model, scaler, PCA, and selected features
joblib.dump(best_rf_model, "trained_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(selected_feature_names, "selected_features.pkl")

print("✅ Model, Scaler, PCA, and Selected Features saved successfully!")

# ------------------------ User Input Prediction ------------------------
# Load saved model and transformers
model = joblib.load("trained_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
selected_features = joblib.load("selected_features.pkl")

# Take user input for the selected 10 features
user_input = {}
for feature in selected_features:
    user_input[feature] = float(input(f"Enter value for {feature}: "))

# Convert input to DataFrame
user_df = pd.DataFrame([user_input])

# Select only the required features
user_selected = user_df[selected_features]

# Scale and apply PCA transformation
user_scaled = scaler.transform(user_selected)
user_pca = pca.transform(user_scaled)

# Predict effort
user_pred = model.predict(user_pca)

print(f"\nPredicted Actual Effort based on user input: {user_pred[0]:.4f}")'''
'''#68 stable rf import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import optuna
from optuna.samplers import TPESampler

# Fix random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load dataset
file_path = os.path.expanduser("~/Downloads/projectdata.csv")  # Update path if needed
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

# Data Augmentation using Gaussian noise
noise_factor = 0.1  # Adjust the noise factor as needed
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
n_components = 5  # Number of principal components
pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X_selected)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_combined, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)

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

# Train the best model
best_rf_model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    random_state=RANDOM_SEED
)
best_rf_model.fit(X_train, y_train)

# Evaluate the tuned model on the test set
y_pred_rf = best_rf_model.predict(X_test)
mae_tuned = mean_absolute_error(y_test, y_pred_rf)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_tuned = r2_score(y_test, y_pred_rf)

print(f"Tuned MAE: {mae_tuned}")
print(f"Tuned RMSE: {rmse_tuned}")
print(f"Tuned R² Score: {r2_tuned}")

# Predict on the original dataset
X_original = df.drop(columns=['Actual_effort', 'Estimated_size', 'Degree_of_standards_usage'])
y_original = df['Actual_effort']

# Standardize features using the same scaler
X_original_scaled = scaler.transform(X_original)

# Select the top 10 features from the original dataset
X_original_selected = X_original_scaled[:, indices]

# Apply PCA to the selected features of the original dataset
X_original_pca = pca.transform(X_original_selected)

# Predict using the tuned model
y_original_pred = best_rf_model.predict(X_original_pca)

# Create a DataFrame to compare actual and predicted values
results_df = pd.DataFrame({
    'Actual_Effort': y_original,
    'Predicted_Effort': y_original_pred
})

# Print the results
print("\nPredictions for the Original Dataset:")
print(results_df.head(10))  # Print the first 10 rows for comparison

# Take user input for prediction
user_input = {}
for col in selected_feature_names:
    value = float(input(f"Enter the value for {col}: "))
    user_input[col] = value

# Create a DataFrame for the user input
user_df = pd.DataFrame(user_input, index=[0])

# Standardize and transform the user input using only the selected features
user_scaled = scaler.transform(X_original)[:, indices]  # Use the same scaling for selected features
user_selected = user_scaled[:1]  # Use the first row for a single input
user_pca = pca.transform(user_selected)

# Predict using the tuned model
user_pred = best_rf_model.predict(user_pca)
print(f"\nPredicted Actual Effort based on user input: {user_pred[0]}")
'''
'''import pandas as pd
import numpy as np
import os
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

# Apply PCA to the selected features (Dynamic adjustment of n_components)
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
# Create a DataFrame to compare actual and predicted values
results_df = pd.DataFrame({
    'Actual_Effort': y_original,
    'Predicted_Effort': y_original_pred
})

# Display the top 10 actual and predicted efforts
print("\nTop 10 Actual and Predicted Efforts:")
print(results_df.head(10))  # Adjust the number if you want more/less rows

# Take user input for prediction
user_input = {}
for col in selected_feature_names:
    value = float(input(f"Enter the value for {col}: "))
    user_input[col] = value

# Create a DataFrame for the user input
user_df = pd.DataFrame(user_input, index=[0])

# Standardize and transform the user input using only the selected features
user_scaled = scaler.transform(X)[:, indices]  # Use the same scaling for selected features
user_selected = user_scaled[:1]  # Use the first row for a single input
user_pca = pca.transform(user_selected)

# Predict using the tuned model
user_pred = best_xgb_model.predict(user_pca)
print(f"\nPredicted Actual Effort based on user input: {user_pred[0]}")'''
'''83 accuracy xbgboost import pandas as pd
import numpy as np
import os
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

# Apply PCA to the selected features (Dynamic adjustment of n_components)
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

# Predict on the original dataset
X_original = df.drop(columns=['Actual_effort', 'Estimated_size', 'Degree_of_standards_usage'])
y_original = df['Actual_effort']

# Standardize features using the same scaler
X_original_scaled = scaler.transform(X_original)

# Select the top 10 features from the original dataset
X_original_selected = X_original_scaled[:, indices]

# Apply PCA to the selected features of the original dataset
X_original_pca = pca.transform(X_original_selected)

# Predict using the tuned model
y_original_pred = best_xgb_model.predict(X_original_pca)

# Create a DataFrame to compare actual and predicted values
results_df = pd.DataFrame({
    'Actual_Effort': y_original,
    'Predicted_Effort': y_original_pred
})

# Display the top 10 actual and predicted efforts
print("\nTop 10 Actual and Predicted Efforts:")
print(results_df.head(10))  # Print the first 10 rows for comparison

# Take user input for prediction
user_input = {}
for col in selected_feature_names:
    value = float(input(f"Enter the value for {col}: "))
    user_input[col] = value

# Create a DataFrame for the user input
user_df = pd.DataFrame(user_input, index=[0])

# Standardize and transform the user input using only the selected features
user_scaled = scaler.transform(X)[:, indices]  # Use the same scaling for selected features
user_selected = user_scaled[:1]  # Use the first row for a single input
user_pca = pca.transform(user_selected)

# Predict using the tuned model
user_pred = best_xgb_model.predict(user_pca)
print(f"\nPredicted Actual Effort based on user input: {user_pred[0]}")'''



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
joblib.dump(list(selected_feature_names), 'selected_features.pkl')

print("✅ Model, scaler, PCA, and selected features saved successfully!")'''
'''#final code 66 import pandas as pd
import numpy as np
import os
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

# Train-Test Split for Evaluation
X_train, X_test, y_train, y_test = train_test_split(X_rfe_scaled, y_combined, test_size=0.2, random_state=RANDOM_SEED)
final_rf_model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = final_rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nEvaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")

# Display top 5 actual and predicted efforts
results_df = pd.DataFrame({'Actual_Effort': y_test, 'Predicted_Effort': y_pred})
print("\nTop 5 Actual and Predicted Efforts:")
print(results_df.head(5))

# Manual Prediction based on user input
print("\nEnter values for the following features:")
user_input = {}
for feature in selected_features:
    while True:
        try:
            user_input[feature] = float(input(f"{feature}: "))
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Preprocess user input for prediction
user_df = pd.DataFrame([user_input])  # Ensure input is a DataFrame

# Reorder columns to match selected_features order
user_df = user_df[selected_features]

# Debugging: Compare transformed manual input with test data
print("Debug - Transformed Test Row (Example):", X_rfe_scaled[1])  # Example of test row

user_scaled = scaler_selected.transform(user_df)  # Scale input with the fitted scaler
print("Debug - Transformed Manual Input:", user_scaled)  # Transformed manual input

# Predict effort
user_pred = final_rf_model.predict(user_scaled)
print(f"\nPredicted Actual Effort based on user input: {user_pred[0]}")'''
''' 72 xgboost import pandas as pd

import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from xgboost import XGBRegressor

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

# Hyperparameter tuning for XGBoost using Optuna
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)

    xgb_model_tuned = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        colsample_bytree=colsample_bytree,
        subsample=subsample,
        random_state=RANDOM_SEED
    )
    scores = cross_val_score(xgb_model_tuned, X_scaled, y_combined, cv=5, scoring='neg_mean_squared_error')
    return -1 * scores.mean()

study = optuna.create_study(direction='minimize', study_name='XGBoost Tuning')
study.optimize(objective, n_trials=50)
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Train final XGBoost model with best hyperparameters
final_xgb_model = XGBRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    colsample_bytree=best_params['colsample_bytree'],
    subsample=best_params['subsample'],
    random_state=RANDOM_SEED
)

# Train-Test Split for Evaluation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_combined, test_size=0.2, random_state=RANDOM_SEED)
final_xgb_model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = final_xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nEvaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")

# Display top 5 actual and predicted efforts
results_df = pd.DataFrame({'Actual_Effort': y_test, 'Predicted_Effort': y_pred})
print("\nTop 5 Actual and Predicted Efforts:")
print(results_df.head(5))

'''
'''#xgb model without pca 
import pandas as pd

import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fix random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load dataset
file_path = os.path.expanduser("~/Downloads/projectdata.csv")  # Update the path if needed
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Handle missing values
# Fill numeric columns with their mean values
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill categorical columns with their mode values
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Label Encoding for categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['Actual_effort'])
y = df['Actual_effort']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# Stage 1: Train Base Models (Random Forest and XGBoost)
rf_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_SEED)

# Use cross-validation to generate meta-features
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
rf_preds = cross_val_predict(rf_model, X_train, y_train, cv=kf, method='predict')
xgb_preds = cross_val_predict(xgb_model, X_train, y_train, cv=kf, method='predict')

# Combine predictions as meta-features
meta_features = np.vstack((rf_preds, xgb_preds)).T

# Stage 2: Train Meta-Model (Linear Regression)
meta_model = LinearRegression()
meta_model.fit(meta_features, y_train)

# Evaluate on the test set
rf_model.fit(X_train, y_train)  # Fit Random Forest on the entire training set
xgb_model.fit(X_train, y_train)  # Fit XGBoost on the entire training set

# Generate meta-features for the test set
rf_test_preds = rf_model.predict(X_test)
xgb_test_preds = xgb_model.predict(X_test)
test_meta_features = np.vstack((rf_test_preds, xgb_test_preds)).T

# Final predictions using the meta-model
final_predictions = meta_model.predict(test_meta_features)

# Evaluate the model
mae = mean_absolute_error(y_test, final_predictions)
rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
r2 = r2_score(y_test, final_predictions)

print(f"\nEvaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")

# Display top 5 actual and predicted efforts
results_df = pd.DataFrame({'Actual_Effort': y_test, 'Predicted_Effort': final_predictions})
print("\nTop 5 Actual and Predicted Efforts:")
print(results_df.head(5))'''

# desh rf and xgboost import pandas as pd
'''import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from imblearn.over_sampling import SMOTE
import os
import pandas as pd
# For data sampling

# Fix random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load dataset
# Load dataset"02.desharnais.c
file_path = os.path.expanduser("~/Downloads/02.desharnais.csv") # Update the path if needed
data = pd.read_csv(file_path)


# Clean column names
data.columns = data.columns.str.strip().str.replace(" ", "_")
print(data.columns.tolist())
# Drop the 'Length' column
data.drop(columns=['Length','id', 'Project'], inplace=True)

# Handle missing values
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Label Encoding for categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Features and target
X = data.drop(columns=['Effort'])
y = data['Effort']

# Data Sampling: Use SMOTE for handling imbalanced data
smote = SMOTE(random_state=RANDOM_SEED)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Standardize the data
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=RANDOM_SEED)

# Define models
rf_model = RandomForestRegressor(random_state=RANDOM_SEED)
xgb_model = XGBRegressor(random_state=RANDOM_SEED)

# Hyperparameter tuning for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, scoring='neg_mean_squared_error',
                              n_jobs=-1)
rf_grid_search.fit(X_train, y_train)
rf_best_model = rf_grid_search.best_estimator_
print(f"Best Random Forest Params: {rf_grid_search.best_params_}")

# Hyperparameter tuning for XGBoost
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.2]
}

xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=5, scoring='neg_mean_squared_error',
                               n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)
xgb_best_model = xgb_grid_search.best_estimator_
print(f"Best XGBoost Params: {xgb_grid_search.best_params_}")

# Evaluate both models
models = {'Random Forest': rf_best_model, 'XGBoost': xgb_best_model}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{model_name} Performance:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R² Score: {r2}")

# Final Effort Calculation
best_model = xgb_best_model if r2_score(y_test, xgb_best_model.predict(X_test)) > r2_score(y_test,
                                                                                           rf_best_model.predict(
                                                                                               X_test)) else rf_best_model
final_effort_predictions = best_model.predict(X_test)

results_df = pd.DataFrame({'Actual_Effort': y_test, 'Predicted_Effort': final_effort_predictions})
print("\nTop 5 Actual and Predicted Efforts:")
print(results_df.head())'''
'''rf 54 import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Fix random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load dataset (replace with your file path)
file_path = os.path.expanduser("~/Downloads/02.desharnais.csv") # Update the path if needed
data = pd.read_csv(file_path)


# Clean column names
data.columns = data.columns.str.strip().str.replace(" ", "_")
print(data.columns.tolist())
# Drop the 'Length' column
data.drop(columns=['Length','id', 'Project'], inplace=True)

# Handle missing values
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Label Encoding for categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Features and target
X = data.drop(columns=['Effort'])
y = data['Effort']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=RANDOM_SEED)

# Random Forest model
rf_model = RandomForestRegressor(random_state=RANDOM_SEED)

# Parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
}

# Grid search for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model and hyperparameters
best_rf_model = grid_search.best_estimator_
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Evaluate the best model
best_rf_model.fit(X_train, y_train)
y_pred = best_rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Metrics for Optimized Random Forest:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")

# Display top 5 actual and predicted values
results_df = pd.DataFrame({'Actual_Effort': y_test, 'Predicted_Effort': y_pred})
print("\nTop 5 Actual and Predicted Efforts:")
print(results_df.head(5))'''
'''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
import os
# Fix random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load dataset
# Load dataset (replace with your file path)
file_path = os.path.expanduser("~/Downloads/02.desharnais.csv") # Update the path if needed
data = pd.read_csv(file_path)


# Clean column names
data.columns = data.columns.str.strip().str.replace(" ", "_")
print(data.columns.tolist())
# Drop the 'Length' column
data.drop(columns=['Length','id', 'Project'], inplace=True)


# Handle missing values
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Label Encoding for categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Define features and target
X = data.drop(columns=['Effort'])
y = data['Effort']

# Check the distribution of target variable before resampling
print("Original Target Distribution:")
print(y.value_counts())

# Use SMOTE for oversampling
smote = SMOTE(random_state=RANDOM_SEED)
X_smote, y_smote = smote.fit_resample(X, y)

# Check the distribution of target variable after resampling
print("Resampled Target Distribution:")
print(y_smote.value_counts())

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_smote)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_smote, test_size=0.2, random_state=RANDOM_SEED)

# Train Random Forest model
rf_model = RandomForestRegressor(random_state=RANDOM_SEED)

# Evaluate using cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Train and evaluate on the test set
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nCross-Validation Results (Train Data):")
print(f"CV Mean Squared Error: {-np.mean(cv_scores):.2f}")
print("\nEvaluation Metrics (Test Data):")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Display top 5 actual and predicted efforts
results_df = pd.DataFrame({'Actual_Effort': y_test, 'Predicted_Effort': y_pred})
print("\nTop 5 Actual and Predicted Efforts:")
print(results_df.head(5))'''
'''import pandas as pd
import numpy as np
import os
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

# Train-Test Split for Evaluation
X_train, X_test, y_train, y_test = train_test_split(X_rfe_scaled, y_combined, test_size=0.2, random_state=RANDOM_SEED)
final_rf_model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = final_rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nEvaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")

# Display top 5 actual and predicted efforts
results_df = pd.DataFrame({'Actual_Effort': y_test, 'Predicted_Effort': y_pred})
print("\nTop 5 Actual and Predicted Efforts:")
print(results_df.head(5))

# Manual Prediction based on user input
print("\nEnter values for the following features:")
user_input = {}
for feature in selected_features:
    while True:
        try:
            user_input[feature] = float(input(f"{feature}: "))
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Preprocess user input for prediction
user_df = pd.DataFrame([user_input])  # Ensure input is a DataFrame

# Reorder columns to match selected_features order
user_df = user_df[selected_features]

# Debugging: Compare transformed manual input with test data
print("Debug - Transformed Test Row (Example):", X_rfe_scaled[1])  # Example of test row

user_scaled = scaler_selected.transform(user_df)  # Scale input with the fitted scaler
print("Debug - Transformed Manual Input:", user_scaled)  # Transformed manual input

# Predict effort
user_pred = final_rf_model.predict(user_scaled)
print(f"\nPredicted Actual Effort based on user input: {user_pred[0]}")
'''
'''import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import os
from openpyxl import load_workbook

# Load the dataset
file_path = os.path.expanduser("~/Downloads/SEERA dataset original raw data - SEERA dataset .csv")
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Keep only required features
features_to_keep = [
     'Object_points',  'Degree_of_software_reuse', 'Programmers_experience_in_programming_language', 'Programmers_capability','Team_size','Dedicated_team_members', 'Daily_working_hours',
    'Actual_effort', 'Actual_duration'
]
df = df[features_to_keep]

# Convert all columns to numeric (if not already)
df = df.apply(pd.to_numeric, errors='coerce')

# Impute missing values with mean for all columns
imputer = SimpleImputer(strategy='mean')
df[:] = imputer.fit_transform(df)

# Round all float columns and convert to int
df = df.round().astype(int)

# Save to Excel
ml_file = "traditionalfeaturesinput.xlsx"
df.to_excel(ml_file, index=False)

# Adjust column widths in Excel
wb = load_workbook(ml_file)
ws = wb.active

for col in ws.columns:
    max_length = 0
    column = col[0].column_letter  # Get column letter (e.g., 'A')
    for cell in col:
        try:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        except:
            pass
    adjusted_width = max(max_length, len(str(col[0].value))) + 2  # Add padding
    ws.column_dimensions[column].width = adjusted_width

wb.save(ml_file)'''
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
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

# Split data into train and test BEFORE scaling and feature selection
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=RANDOM_SEED)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection using RFE with RandomForest on TRAIN ONLY
rf_model = RandomForestRegressor(random_state=RANDOM_SEED)
rfe = RFE(estimator=rf_model, n_features_to_select=10)
X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
X_test_rfe = rfe.transform(X_test_scaled)

selected_features = X.columns[rfe.get_support()]
print(f"Selected Features: {list(selected_features)}")

# Save selected features
with open("selected_features.pkl", "wb") as f:
    pickle.dump(list(selected_features), f)

# Reinitialize scaler for selected features (optional step if needed later)
scaler_selected = StandardScaler()
X_train_final = scaler_selected.fit_transform(X_train_rfe)
X_test_final = scaler_selected.transform(X_test_rfe)

# Hyperparameter tuning using Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
    }
    model = RandomForestRegressor(**params, random_state=RANDOM_SEED)
    scores = cross_val_score(model, X_train_final, y_train, cv=5, scoring='neg_mean_squared_error')
    return -1 * scores.mean()

study = optuna.create_study(direction='minimize', study_name='RF_RFE_Tuning')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Train final Random Forest model
final_rf_model = RandomForestRegressor(**best_params, random_state=RANDOM_SEED)
final_rf_model.fit(X_train_final, y_train)

# Save model and scaler
with open("random_forest_model.pkl", "wb") as model_file:
    pickle.dump(final_rf_model, model_file)
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler_selected, scaler_file)

print("Model and scaler saved successfully.")

# ==================
# Evaluation Metrics
# ==================
y_pred = final_rf_model.predict(X_test_final)

# Compute Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmsle = np.sqrt(mean_squared_log_error(y_test, np.maximum(0, y_pred)))  # Ensure non-negative
r2 = r2_score(y_test, y_pred)

# Print Evaluation Metrics
print("\n📊 Evaluation Metrics on Test Set:")
print(f"MAE   : {mae:.4f}")
print(f"MSE   : {mse:.4f}")
print(f"RMSE  : {rmse:.4f}")
print(f"RMSLE : {rmsle:.4f}")
print(f"R²    : {r2:.4f}")



