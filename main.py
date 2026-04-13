import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ✅ Configuration & Path Setup
DATA_PATH = "data/credit_data.csv"
MODEL_PATH = "models/credit_model.pkl"
OUTPUT_DIR = "outputs"
os.makedirs("models", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_visuals():
    """Sets a global dark theme for professional charts."""
    plt.style.use('dark_background')
    sns.set_palette("Reds_r")

def generate_analytics(df, X_test, y_test, models):
    """Generates all PNG assets for the Streamlit Dashboard."""
    print("📊 Generating Visual Analytics...")

    # 1. Accuracy Comparison Chart
    plt.figure(figsize=(10, 6))
    acc_scores = {name: accuracy_score(y_test, m.predict(X_test)) for name, m in models.items()}
    sns.barplot(x=list(acc_scores.keys()), y=list(acc_scores.values()))
    plt.title("Algorithm Performance Comparison", fontsize=15, pad=20)
    plt.ylabel("Accuracy Score")
    plt.ylim(0, 1.1)
    plt.savefig(f"{OUTPUT_DIR}/accuracy_comparison.png", bbox_inches='tight', transparent=True)
    plt.close()

    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap="Reds", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix", fontsize=15, pad=20)
    plt.savefig(f"{OUTPUT_DIR}/heatmap.png", bbox_inches='tight')
    plt.close()

    # 3. Risk Distribution Bar
    plt.figure(figsize=(8, 6))
    df['Risk'].value_counts().plot(kind='bar', color=['#444444', '#ff0033'])
    plt.title("Class Balance: High vs Low Risk", fontsize=14)
    plt.xticks([0, 1], ['Low Risk', 'High Risk'], rotation=0)
    plt.savefig(f"{OUTPUT_DIR}/risk_distribution.png", bbox_inches='tight')
    plt.close()

    # 4. Decision Tree Logic Visual
    plt.figure(figsize=(20, 10))
    dt_viz = DecisionTreeClassifier(max_depth=3)
    dt_viz.fit(df.drop('Risk', axis=1), df['Risk'])
    plot_tree(dt_viz, feature_names=df.columns[:-1], class_names=['Low', 'High'], 
              filled=True, rounded=True, fontsize=10)
    plt.title("Model Decision Logic (Inference Path)", fontsize=18)
    plt.savefig(f"{OUTPUT_DIR}/decision_tree.png", bbox_inches='tight')
    plt.close()

def main():
    setup_visuals()
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found. Please ensure the dataset exists.")
        return

    # --- Step 1: Data Preparation ---
    print("📂 Loading Dataset...")
    df = pd.read_csv(DATA_PATH)
    X = df.drop('Risk', axis=1)
    y = df['Risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Step 2: Model Training Benchmarks ---
    print("⚙️ Training Models...")
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"   ✅ {name} Trained.")

    # --- Step 3: Generate Visuals for Dashboard ---
    generate_analytics(df, X_test, y_test, models)

    # --- Step 4: Export Production Model ---
    # We use Random Forest for the final export due to its stability
    print(f"💾 Saving Production Model to {MODEL_PATH}...")
    joblib.dump(models["Random Forest"], MODEL_PATH)

    print("\n🚀 SUCCESS: Project artifacts generated.")
    print("   Run 'streamlit run app/main_app.py' to launch the dashboard.")

if __name__ == "__main__":
    main()