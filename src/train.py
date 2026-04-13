import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def train_model():
    # Load data
    df = pd.read_csv("data/credit_data.csv")
    X = df.drop("Risk", axis=1)
    y = df["Risk"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Generate Accuracy Comparison
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }
    
    accuracies = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        accuracies[name] = accuracy_score(y_test, m.predict(X_test))

    # Save Accuracy Graph (Dark Theme)
    plt.style.use('dark_background')
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette="Reds_r")
    plt.title("Model Accuracy Comparison")
    plt.savefig("outputs/accuracy_comparison.png", transparent=True)

    # 2. Save Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="Reds", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.savefig("outputs/heatmap.png")

    # 3. Save Final Model (Using Random Forest for production)
    final_model = RandomForestClassifier(n_estimators=100)
    final_model.fit(X, y)
    joblib.dump(final_model, "models/credit_model.pkl")
    
    # 4. Save Decision Tree Logic Visual
    dt_model = DecisionTreeClassifier(max_depth=3)
    dt_model.fit(X, y)
    plt.figure(figsize=(15, 10))
    plot_tree(dt_model, feature_names=X.columns, class_names=["Low", "High"], filled=True)
    plt.savefig("outputs/decision_tree.png")

    print("✅ Training Complete. Model and Analytics saved to /models and /outputs.")

if __name__ == "__main__":
    train_model()