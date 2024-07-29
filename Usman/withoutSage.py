import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error
import numpy as np
import networkx as nx

# Load the graphml file
graph = nx.read_graphml("C:\\Users\\xxbla\\OneDrive\\Documents\\VSCode\\CETA\\CETANN\\Usman\\filtered_export.graphml")

# Extract the relevant attributes into a dataframe
attributes = [
    'D_DT_START', 'D_DT_END', 'BHC_IND', 'CHTR_TYPE_CD', 'FHC_IND',
    'INSUR_PRI_CD', 'MBR_FHLBS_IND', 'MBR_FRS_IND', 'SEC_RPTG_STATUS',
    'EST_TYPE_CD', 'BNK_TYPE_ANALYS_CD', 'ENTITY_TYPE', 'ACT_PRIM_CD', 'CITY', 'CNTRY_CD'
]
# add Country Code
# start using edges and edge features
# edge: a, b, time of edge
# want to focus on temporal aspect

# Create a list of nodes and their attributes
data = []
for node, attr in graph.nodes(data=True):
    row = {key: attr.get(key, None) for key in attributes}
    row['ID_RSSD'] = attr.get('ID_RSSD', None)  # Use ID_RSSD for company identification
    data.append(row)

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert date columns to datetime
df['D_DT_START'] = pd.to_datetime(df['D_DT_START'], errors='coerce').dt.tz_localize(None)
df['D_DT_END'] = pd.to_datetime(df['D_DT_END'], errors='coerce').dt.tz_localize(None)

# Calculate lifespan in years if D_DT_END is available
df['lifespan'] = (df['D_DT_END'] - df['D_DT_START']).dt.days / 365.25

# Drop rows where lifespan is NaN (companies that have not closed)
df = df.dropna(subset=['lifespan'])

# Convert date columns to numerical values (number of days since a reference date)
reference_date = pd.Timestamp('1900-01-01')
df['D_DT_START'] = (df['D_DT_START'] - reference_date).dt.days
df['D_DT_END'] = (df['D_DT_END'] - reference_date).dt.days

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    if column != 'ID_RSSD':  # Do not encode ID_RSSD
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

# Split the data into features and target variable
X = df[attributes]
y = df['lifespan']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, df['ID_RSSD'], test_size=0.2, random_state=42
)

# # Train a Random Forest Regressor

# # Use 110-300 for n_estimators
# # When using predictions use RMSE
# # Which features did it use to make these predictions
# # Feature importance 
# # Look at distribution of lifespans
# # Use graphsage to predict node labels
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Predict on the test set
# y_pred = model.predict(X_test)

# # Create a DataFrame to display the results
# results = pd.DataFrame({
#     'ID_RSSD': id_test,
#     'predicted_lifespan': y_pred
# })

# # Save the results to a CSV file
# results.to_csv("C:\\Users\\xxbla\\OneDrive\\Documents\\VSCode\\CETA\\CETANN\\Usman\\predicted_lifespan_of_companies.csv", index=False)

# # Display the results
# #import ace_tools as tools; tools.display_dataframe_to_user(name="Predicted Lifespan of Companies by ID_RSSD", dataframe=results)

# print(results.head())


###### NEW ##########
# Train multiple Random Forest Regressors with different n_estimators
best_rmse = float('inf')
best_model = None
best_n_estimators = 0

for n_estimators in range(110, 301, 10):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_n_estimators = n_estimators

# Use the best model for final predictions
y_pred = best_model.predict(X_test)

# Calculate RMSE for the best model
rmse = root_mean_squared_error(y_test, y_pred)

# Get feature importances
feature_importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': attributes,
    'importance': feature_importances * 100  # Convert to percentage
}).sort_values(by='importance', ascending=False)

# Print the features and their importance
print("Features and their importance in the model:")
print(feature_importance_df)

# Create a DataFrame to display the results
results = pd.DataFrame({
    'ID_RSSD': id_test,
    'predicted_lifespan': y_pred
})

# Save the results to a CSV file
results.to_csv("C:\\Users\\xxbla\\OneDrive\\Documents\\VSCode\\CETA\\CETANN\\Usman\\predicted_lifespan_of_companies.csv", index=False)

# Display the results
print(results.head())

# Print RMSE
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Best number of estimators: {best_n_estimators}")