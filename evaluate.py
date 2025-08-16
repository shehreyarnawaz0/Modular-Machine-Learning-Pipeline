import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate ML model with classification report, metrics, confusion matrix, and feature importance if available.
    """
    y_pred = model.predict(X_test)

    print(f"\n📊 Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred))

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
    }
    print("Metrics:", metrics)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Feature importance (if available)
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(6, 4))
        sns.barplot(x=model.feature_importances_, y=range(len(model.feature_importances_)))
        plt.title(f"Feature Importance - {model_name}")
        plt.show()
