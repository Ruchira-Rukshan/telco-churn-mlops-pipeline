import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")

st.title("Customer Churn Analytics")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # Basic preprocessing for EDA
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)
        return df
    return None

df = load_data()

if df is not None:
    # KPI Cards
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_customers = len(df)
    churned_customers = len(df[df['Churn'] == 'Yes'])
    churn_rate = (churned_customers / total_customers) * 100
    avg_monthly_charges = df['MonthlyCharges'].mean()
    avg_tenure = df['tenure'].mean()
    
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned_customers:,}")
    col3.metric("Churn Rate", f"{churn_rate:.1f}%")
    col4.metric("Avg Monthly Charges", f"${avg_monthly_charges:.2f}")
    col5.metric("Avg Tenure", f"{avg_tenure:.1f} months")
    
    st.markdown("---")
    
    # Visualizations
    st.subheader("Visual Analytics")
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        # Churn Distribution
        fig1 = px.pie(df, names='Churn', title="Churn Distribution", color='Churn',
                      color_discrete_map={'Yes':'#ef553b', 'No':'#636efa'})
        st.plotly_chart(fig1, use_container_width=True)
        
        # Contract Type vs Churn
        fig2 = px.histogram(df, x="Contract", color="Churn", barmode="group",
                            title="Contract Type vs Churn")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Payment Method vs Churn
        fig3 = px.histogram(df, x="PaymentMethod", color="Churn", barmode="group",
                            title="Payment Method vs Churn")
        st.plotly_chart(fig3, use_container_width=True)
        
    with col_v2:
        # Tenure Distribution
        fig4 = px.histogram(df, x="tenure", color="Churn", nbins=20,
                            title="Tenure Distribution")
        st.plotly_chart(fig4, use_container_width=True)
        
        # Monthly Charges Distribution
        fig5 = px.histogram(df, x="MonthlyCharges", color="Churn", nbins=20,
                            title="Monthly Charges Distribution")
        st.plotly_chart(fig5, use_container_width=True)
        
        # Internet Service vs Churn
        fig6 = px.histogram(df, x="InternetService", color="Churn", barmode="group",
                            title="Internet Service vs Churn")
        st.plotly_chart(fig6, use_container_width=True)
        
    # Correlation Matrix (Numerical)
    st.subheader("Correlation Heatmap")
    df_numeric = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    df_numeric['Churn_Numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})
    corr = df_numeric.corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlations")
    st.plotly_chart(fig_corr, use_container_width=True)

else:
    st.warning("Dataset not found. Please ensure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is in the 'data/' folder to view analytics.")
