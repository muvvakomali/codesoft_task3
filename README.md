import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
data = pd.read_csv(r"C:\Users\komal\Downloads\bank.csv")  # Updated with raw string to fix path error

# Print column names to verify the target column
print("Dataset Columns:", data.columns)

# Handle categorical data
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target (adjust target column if necessary)
X = data.drop('Exited', axis=1)  # Changed from 'Churn' to 'Exited'
y = data['Exited']  # 0 = retained, 1 = churned

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)

# Evaluation
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_preds))
print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_preds))
print("\nGradient Boosting Accuracy:", accuracy_score(y_test, gb_preds))
print("\nClassification Report (Gradient Boosting):\n", classification_report(y_test, gb_preds))
