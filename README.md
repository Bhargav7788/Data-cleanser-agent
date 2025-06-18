# 🧹 CleanGenie – Smart Data Cleaning App

**CleanGenie** is a real-time, interactive data cleaning tool built with Streamlit. It allows you to upload any dataset, visualize, clean, and download it with ease — no coding required.


## 🔧 Features

📄 **Upload CSV** and preview data instantly
🧼 **Handle Missing Values** (mean, median, mode)
🗑️ **Drop High-Null Columns** and **Duplicates**
🧹 **Standardize Text** fields
🚨 **Remove Outliers** (Z-Score / IQR)
📊 **Cleaning Summary** tab with before/after stats
📈 **Visualizations:** Null Heatmap + Boxplots
📥 **Export Cleaned Data** in CSV + Excel

## 🛠️ Tech Stack

**Frontend/UI:** Streamlit
**Backend:** Python (Pandas, NumPy)
**Visualization:** Seaborn, Matplotlib
**Export:** XlsxWriter


## 🚀 Run Locally
https://github.com/Bhargav7788/Data-cleanser-agent/
```bash
git clone 
cd CleanGenie
pip install -r requirements.txt
streamlit run app.py
