import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px

# ✅ Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict

# 🎨 PREMIUM UI CONFIGURATION
st.set_page_config(page_title="RiskIQ | Credit Intelligence", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0d0d0d; color: white; }
    h1 { color: #ff0033; text-align: center; font-weight: 800; }
    .stButton>button {
        background-color: #ff0033;
        color: white;
        border-radius: 8px;
        width: 100%;
        font-weight: bold;
        border: none;
    }
    .stMetric { background-color: #1a1a1a; padding: 15px; border-radius: 10px; border-left: 5px solid #ff0033; }
</style>
""", unsafe_allow_html=True)

# 📂 LOAD BASE DATASET FOR COMPARISON
@st.cache_data
def load_data():
    return pd.read_csv("data/credit_data.csv")

df = load_data()

# 🚀 SIDEBAR - GLOBAL ANALYTICS
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/584/584011.png", width=100)
    st.title("RiskIQ Analytics")
    st.markdown("---")
    
    view_mode = st.radio("Navigation", ["Risk Calculator", "Model Transparency", "Dataset Explorer"])
    
    st.markdown("---")
    st.info("This model uses a Random Forest Classifier trained on synthetic financial data.")

# --- PAGE 1: RISK CALCULATOR ---
if view_mode == "Risk Calculator":
    st.title("💳 Credit Risk Intelligence")
    st.subheader("Real-time automated credit evaluation")

    # 📥 INPUT SECTION
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Client Age", 18, 65, 30)
            income = st.number_input("Annual Income ($)", 20000, 200000, 55000)
        with col2:
            credit_score = st.slider("Credit Score", 300, 850, 650)
            employment_years = st.slider("Years of Employment", 0, 30, 5)
        with col3:
            utilization = st.slider("Utilization (%)", 0, 100, 30)
            existing_loans = st.slider("Active Loans", 0, 5, 1)

        c4, c5 = st.columns(2)
        with c4:
            payment_history = st.selectbox("Payment History", [1, 0], format_func=lambda x: "Good (No Late Payments)" if x==1 else "Bad (Recent Late Payments)")
        with c5:
            default_history = st.selectbox("Default History", [0, 1], format_func=lambda x: "No Prior Defaults" if x==0 else "Has Prior Default")

    if st.button("🚀 EXECUTE RISK ANALYSIS"):
        input_data = {
            "Age": age, "Income": income, "CreditScore": credit_score,
            "Utilization": utilization, "PaymentHistory": payment_history,
            "ExistingLoans": existing_loans, "DefaultHistory": default_history,
            "EmploymentYears": employment_years
        }

        result = predict(input_data)
        risk_val = 1 if "High" in result else 0

        st.markdown("---")
        
        # 🎯 RESULTS & METRICS
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if risk_val == 0:
                st.success(f"### {result}")
                st.progress(25)
            else:
                st.error(f"### {result}")
                st.progress(85)
            
            # Comparison Metrics
            avg_income = df['Income'].mean()
            st.metric("Income vs Average", f"${income:,}", f"{((income-avg_income)/avg_income)*100:.1f}%")

        with res_col2:
            # INTERACTIVE PLOTLY SCATTER
            fig = px.scatter(df, x="Income", y="CreditScore", color="Risk", 
                             color_discrete_map={0: '#00ffcc', 1: '#ff0033'},
                             template="plotly_dark", title="Client Positioning in Market")
            fig.add_scatter(x=[income], y=[credit_score], mode="markers", 
                            marker=dict(size=18, color="white", symbol="star"), name="This Client")
            st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: MODEL TRANSPARENCY ---
elif view_mode == "Model Transparency":
    st.title("🔍 Model Explainability")
    output_path = "outputs"
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.image(f"{output_path}/accuracy_comparison.png", caption="Model Performance")
        st.image(f"{output_path}/heatmap.png", caption="Feature Correlation")
    
    with col_b:
        st.image(f"{output_path}/decision_tree.png", caption="Decision Logic (Inference Path)")

# --- PAGE 3: DATASET EXPLORER ---
elif view_mode == "Dataset Explorer":
    st.title("📊 Training Data Overview")
    
    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(df, x="CreditScore", color="Risk", barmode="overlay", template="plotly_dark")
        st.plotly_chart(fig_hist)
    with c2:
        fig_box = px.box(df, x="Risk", y="Utilization", color="Risk", template="plotly_dark")
        st.plotly_chart(fig_box)
    
    st.dataframe(df.head(10), use_container_width=True)

# 📄 FOOTER
st.markdown("---")
st.caption("🚀 RiskIQ System v2.0 | Powered by Random Forest Intelligence")