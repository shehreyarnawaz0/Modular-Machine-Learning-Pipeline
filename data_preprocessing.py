import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(data_url: str):
    """
    Load Titanic dataset, preprocess features, and return train/test splits.
    """
    df = pd.read_csv(data_url)

    # Drop irrelevant columns
    df = df.drop(["Name", "Ticket", "Cabin"], axis=1)

    # Fill missing values
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Encode categorical variables
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    # Features and target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
