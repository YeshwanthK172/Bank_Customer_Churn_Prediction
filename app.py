import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="FinSafe | Churn Analytics",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling for Premium Look ---
# --- Custom Styling for Premium Look ---
st.markdown("""
    <style>
    /* Main container tweaks */
    .main {
        background-color: #f8f9fa;
    }
    /* Clean headers */
    h1 {
        color: #1E3A8A;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    /* Stat cards style */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Model Pipeline ---
@st.cache_resource
def load_model():
    return joblib.load('best_churn_pipeline.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- Header Section ---
st.title("📊 Customer Retention & Churn Analytics")
st.markdown("Assess customer attrition risk profiles using our synchronized predictive pipeline engine.")
st.write("---")

# --- UI Layout with Organized Columns ---
st.subheader("👤 Customer Profile Input")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("##### **Demographics**")
    country = st.selectbox("🌎 Country / Region", options=["France", "Germany", "Spain"])
    gender = st.selectbox("🚻 Gender", options=["Male", "Female"])
    age = st.slider("📅 Age (Years)", min_value=18, max_value=100, value=38)
    credit_score = st.slider("💳 Credit Score", min_value=300, max_value=850, value=650)
    tenure = st.slider("⏳ Tenure (Years with Bank)", min_value=0, max_value=10, value=5)

with col2:
    st.markdown("##### **Account & Financials**")
    balance = st.number_input("💰 Account Balance ($)", min_value=0.0, value=60000.0, step=1000.0, format="%.2f")
    estimated_salary = st.number_input("💵 Estimated Annual Salary ($)", min_value=0.0, value=75000.0, step=1000.0, format="%.2f")
    products_number = st.radio("📦 Number of Products Active", options=[1, 2, 3, 4], horizontal=True)
    
    st.markdown("##### **Engagement Status**")
    credit_card = st.checkbox("💳 Has Active Credit Card", value=True)
    active_member = st.checkbox("🟢 Is Active Engagement Member", value=True)

# Map UI checkbox boolean values to binary inputs
credit_card_val = 1 if credit_card else 0
active_member_val = 1 if active_member else 0

st.write("---")

# --- Prediction & Result Processing ---
if st.button("🔮 Run Churn Diagnostics", type="primary", use_container_width=True):
    
    # Feature Engineering (Matching your exact model structure)
    salary_balance_ratio = np.nan if balance == 0 else (estimated_salary / balance)
    balance_per_product = balance / products_number if products_number > 0 else 0.0

    bins = [0, 25, 35, 45, 55, 65, 100]
    labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
    age_group = pd.cut([age], bins=bins, labels=labels)[0]
    
    tenure_bucket = pd.cut([tenure], bins=[-1, 0, 2, 5, 10, 100], labels=['0', '1-2', '3-5', '6-10', '10+'])[0]
    high_balance = 1 if balance > 50000.0 else 0

    # DataFrame creation
    sample_df = pd.DataFrame([{
        'credit_score': credit_score,
        'country': country,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'products_number': products_number,
        'credit_card': credit_card_val,
        'active_member': active_member_val,
        'estimated_salary': estimated_salary,
        'salary_balance_ratio': salary_balance_ratio,
        'balance_per_product': balance_per_product,
        'age_group': age_group,
        'tenure_bucket': tenure_bucket,
        'high_balance': high_balance
    }])
    
    # Sync missing values cleanup steps
    sample_df['balance'] = sample_df['balance'].replace(0, np.nan)
    sample_df['salary_balance_ratio'] = sample_df['salary_balance_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Run the processing and prediction
    try:
        prediction = model.predict(sample_df)[0]
        prediction_proba = model.predict_proba(sample_df)[0][1]
        
        st.subheader("🎯 Diagnostic Risk Summary")
        
        # UI Presentation Cards based on outputs
        if prediction == 1:
            st.error(f"🚨 **High Attrition Risk Detected**")
            
            # Interactive metric columns
            c1, c2 = st.columns(2)
            with c1:
                st.metric(label="Churn Probability", value=f"{prediction_proba:.1%}", delta="High Risk", delta_color="inverse")
            with c2:
                st.metric(label="Customer Status", value="Likely to Leave")
                
            st.markdown("""
                > 💡 **Recommended Retention Steps:**
                > * Propose custom loyalty discounts or zero-fee options on current banking accounts.
                > * Reach out with premium relationship management support to re-engage active product usage.
            """)
        else:
            st.success(f"💚 **Low Risk / Loyal Profile Profile**")
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric(label="Retention Score", value=f"{1 - prediction_proba:.1%}", delta="Healthy Profile")
            with c2:
                st.metric(label="Customer Status", value="Stable / Safe")
                
            st.markdown("> 👍 **Status Evaluation:** This user shows high structural interaction features. Maintain regular milestone update tracking sequences.")
            
    except Exception as e:
        st.error(f"An unexpected inference mismatch trace surfaced: {e}")