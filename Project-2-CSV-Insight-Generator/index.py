import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="CSV Insight Generator")

st.title("CSV Insight Generator")
st.markdown("Upload a CSV file to automatically clean data, genrate summaries, & detect trends")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        df.columns = df.columns.str.strip()
        for col in df.columns:
            if df[col].dtype == 'object':
                clean_col = df[col].astype(str).str.replace(r'[^0-9.-]', '', regex=True)
                converted = pd.to_numeric(clean_col, errors='coerce')
                if not converted.isna().all():
                    df[col] = converted

        df = df.dropna(axis=1, how='all')
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Data Summary:")
                st.write(df[numeric_cols].describe().round(2))

            with col2:
                st.subheader("Trend Analysis")
                target_col = numeric_cols[0]
                y = df[target_col].dropna().values
                
                if len(y) > 1:
                    X = np.arange(len(y)).reshape(-1, 1)
                    model = LinearRegression().fit(X, y)
                    slope = model.coef_[0]
                    
                    if slope > 0:
                        st.success(f"UPWARD trend in {target_col}")
                    else:
                        st.error(f"DOWNWARD trend in {target_col}")
                    st.caption(f"Slope Coefficient: {slope:.4f}")
                else:
                    st.info("Not enough data points for trend detection.")
            st.divider()
            st.subheader("Visual Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(df[target_col].dropna(), kde=True, color='teal', ax=ax)
            ax.set_title(f"Distribution of {target_col}")
            st.pyplot(fig)

        else:
            st.warning("No numeric columns found in this CSV")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Waiting fo CSV upload")