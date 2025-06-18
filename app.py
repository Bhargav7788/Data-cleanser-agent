import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import cleaner.fix as fix

st.set_page_config(page_title="CleanGenie: Smart Data Cleaning App", layout="wide")
st.title("🧹 CleanGenie: Smart Data Cleaning Agent")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    original_df = df.copy()
    st.success("✅ File uploaded successfully!")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Raw Data", "🧼 Clean Data", "📥 Download", "📊 Summary", "📈 Data Visuals"
    ])

    # Tab 1: Raw Data
    with tab1:
        st.header("📄 Raw Data Preview")
        st.dataframe(original_df)

    # Tab 2: Clean Data
    with tab2:
        st.header("🧼 Clean Data Options")

        if st.checkbox("Impute Missing Values"):
            strategy = st.radio("Choose Strategy", ["mean", "median", "mode"], horizontal=True)
            df = fix.impute_missing_values(df, strategy)

        if st.checkbox("Drop High Null Columns"):
            threshold = st.slider("Null Threshold", 0.0, 1.0, 0.5)
            df = fix.drop_high_null_columns(df, threshold)

        if st.checkbox("Standardize Text Columns"):
            df = fix.standardize_text(df)

        if st.checkbox("Drop Duplicates"):
            df = fix.drop_duplicates(df)

        outlier_method = st.radio("Outlier Detection Method", ["None", "Z-Score", "IQR"], horizontal=True)
        if st.button("Remove Outliers"):
            if outlier_method == "Z-Score":
                df = fix.remove_outliers_zscore(df)
                st.success("✅ Removed outliers using Z-Score.")
            elif outlier_method == "IQR":
                df = fix.remove_outliers_iqr(df)
                st.success("✅ Removed outliers using IQR.")
            else:
                st.info("ℹ️ No outlier removal applied.")

        st.subheader("🔎 Cleaned Data Preview")
        st.dataframe(df)

    # Tab 3: Download
    with tab3:
        st.header("📥 Download Cleaned Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", csv, "cleaned_data.csv", "text/csv")

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Cleaned Data')
        st.download_button("⬇️ Download Excel", output.getvalue(), "cleaned_data.xlsx")

    # Tab 4: Summary
    with tab4:
        st.header("📊 Cleaning Summary")
        before_rows, before_cols = original_df.shape
        after_rows, after_cols = df.shape

        nulls_before = original_df.isnull().sum().sum()
        nulls_after = df.isnull().sum().sum()
        dups_removed = before_rows - original_df.drop_duplicates().shape[0]

        st.write(f"**🧮 Rows:** {before_rows} ➝ {after_rows}")
        st.write(f"**🧾 Columns:** {before_cols} ➝ {after_cols}")
        st.write(f"**❌ Nulls:** {nulls_before} ➝ {nulls_after}")
        st.write(f"**🗑️ Duplicates Removed:** {dups_removed}")

        st.metric("Null Reduction", f"{100 * (nulls_before - nulls_after) / (nulls_before + 1):.2f}%")
        st.metric("Data Retained", f"{100 * after_rows / (before_rows + 1):.2f}%")

    # Tab 5: Visuals
    with tab5:
        st.header("📈 Data Visuals")

        st.subheader("🔍 Null Value Heatmap")
        fig_null, ax = plt.subplots()
        sns.heatmap(df.isnull(), cbar=False, cmap="YlOrRd", ax=ax)
        st.pyplot(fig_null)

        st.subheader("📦 Boxplots for Numeric Columns")
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(num_cols) == 0:
            st.warning("⚠️ No numeric columns available.")
        else:
            selected_col = st.selectbox("Choose a column", num_cols)
            fig_box, ax2 = plt.subplots()
            sns.boxplot(x=df[selected_col], ax=ax2, color="skyblue")
            st.pyplot(fig_box)
