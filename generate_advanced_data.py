import pandas as pd
import numpy as np
import os
from scipy.stats import skewnorm

def generate_advanced_credit_data(num_samples=2000):
    """
    Generates a high-fidelity synthetic dataset with non-linear relationships,
    skewed distributions, and intentional noise.
    """
    np.random.seed(42)

    # 1. GENERATING SKEWED DISTRIBUTIONS (More realistic than uniform)
    # Most people have average income, few have very high (Right Skewed)
    income = skewnorm.rvs(a=10, loc=20000, scale=40000, size=num_samples).clip(15000, 250000)
    
    # Age distribution
    age = np.random.normal(loc=40, scale=12, size=num_samples).astype(int).clip(18, 75)
    
    # 2. MULTIVARIATE RELATIONSHIPS (Features depend on each other)
    # Credit Score depends on Age and Income (with some randomness)
    base_score = 400 + (age * 2) + (income / 1000)
    credit_score = base_score + np.random.normal(0, 50, size=num_samples)
    credit_score = credit_score.clip(300, 850).astype(int)

    # Utilization usually higher for lower incomes
    util_base = 80 - (income / 2500)
    utilization = util_base + np.random.normal(0, 15, size=num_samples)
    utilization = utilization.clip(5, 98).astype(float)

    # Employment years can't exceed (age - 18)
    employment_years = np.array([np.random.randint(0, max(1, a - 18)) for a in age])

    # 3. CATEGORICAL PROBABILITIES
    # Higher income/age = lower probability of bad history
    history_prob = (credit_score / 850) * 0.9
    payment_history = np.array([np.random.choice([1, 0], p=[p, 1-p]) for p in history_prob])
    
    default_history = np.random.choice([0, 1], size=num_samples, p=[0.92, 0.08])

    # 4. COMPLEX TARGET LOGIC (Non-Linear Risk)
    # Risk is not just a sum; it's a combination of thresholds
    risk_factor = (
        (700 - credit_score) * 1.5 + 
        (utilization * 2.0) + 
        (25 - employment_years) * 5 +
        (1 - payment_history) * 250
    )
    
    # Add "Systemic Noise" (Unexpected defaults/successes)
    noise = np.random.normal(0, 100, size=num_samples)
    risk_factor += noise

    # Define Risk: 1 = High Risk, 0 = Low Risk
    # We use a dynamic threshold to ensure a 20-30% default rate
    threshold = np.percentile(risk_factor, 72)
    risk = (risk_factor >= threshold).astype(int)

    # 5. INJECTING OUTLIERS (Professional Touch)
    # High income people who are high risk (rare but exists)
    for _ in range(int(num_samples * 0.02)):
        idx = np.random.randint(0, num_samples)
        df_temp_income = 200000
        df_temp_risk = 1 # Force high risk for a high earner to test model robustness

    # 6. FINAL ASSEMBLY
    df = pd.DataFrame({
        "Age": age,
        "Income": np.round(income, -2),
        "CreditScore": credit_score,
        "Utilization": np.round(utilization, 1),
        "PaymentHistory": payment_history,
        "ExistingLoans": np.random.randint(0, 6, size=num_samples),
        "DefaultHistory": default_history,
        "EmploymentYears": employment_years,
        "Risk": risk
    })

    # Save
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/credit_data.csv", index=False)
    
    print("💎 Advanced Synthetic Dataset Created.")
    print(f"📊 Summary Statistics:\n{df['Risk'].value_counts(normalize=True)}")
    return df

if __name__ == "__main__":
    generate_advanced_credit_data(2500)