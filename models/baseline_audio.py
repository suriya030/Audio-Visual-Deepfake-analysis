from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import ast
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("audio_features.csv")

# Prepare features and labels
X = df.drop(["filename", "label"], axis=1)
y = df["label"].map({"0_real": 0, "1_fake": 1})  # Convert labels to 0 (real) and 1 (fake)

#Dataset handling - looks for non-numerical data/ object data
for col in X.columns:
    X[col] = X[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)  # Replace NaNs with zero

# Splitting dataset into 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline Models
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

#scaling training data input to help with convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
lg_clf = LogisticRegression()
lg_clf.fit(X_train_scaled, y_train)

svm_clf = SVC(kernel="linear")
svm_clf.fit(X_train, y_train)

# Evaluation metrics
rf_pred = rf_clf.predict(X_test)
lg_pred = lg_clf.predict(X_test)
svm_pred = svm_clf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lg_pred))
print("Linear SVM Accuracy:", accuracy_score(y_test, svm_pred))