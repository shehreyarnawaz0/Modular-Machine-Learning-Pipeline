import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from data_preprocessing import load_and_preprocess
from evaluate import evaluate_model

# Paths
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load preprocessed data
X_train, X_test, y_train, y_test = load_and_preprocess(DATA_URL)

# Define models & hyperparameters
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {"C": [0.01, 0.1, 1, 10], "solver": ["liblinear"]}
    ),
    "Decision Tree": (
        DecisionTreeClassifier(),
        {"max_depth": [3, 5, 7, None], "criterion": ["gini", "entropy"]}
    ),
    "Random Forest": (
        RandomForestClassifier(),
        {"n_estimators": [50, 100], "max_depth": [3, 5, None]}
    )
}

# Train & evaluate
for name, (model, params) in models.items():
    print(f"\n🔍 Training {name}...")
    grid = GridSearchCV(model, params, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"✅ Best Params for {name}: {grid.best_params_}")

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{name.replace(' ', '_')}.pkl")
    joblib.dump(best_model, model_path)

    # Evaluate
    evaluate_model(best_model, X_test, y_test, name)
