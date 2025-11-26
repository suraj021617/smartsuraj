import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="4D Results Viewer", layout="centered")
st.title("📅 4D Results by Date")

# Load CSV
df = pd.read_csv("4d_results_history.csv")

# ✅ Step 1: Extract actual date from 'draw_info' column using regex
def extract_date(info):
    match = re.search(r"(\d{2}-\d{2}-\d{4})", str(info))
    return match.group(1) if match else None

df["extracted_date"] = df["draw_info"].apply(extract_date)
df["extracted_date"] = pd.to_datetime(df["extracted_date"], format="%d-%m-%Y", errors="coerce")
df = df.dropna(subset=["extracted_date"])
df["only_date"] = df["extracted_date"].dt.date

# ✅ UI: Date Picker
selected_date = st.date_input("Select a Date to View Results")

# ✅ Filter results
filtered = df[df["only_date"] == selected_date]

# ✅ Show results
if not filtered.empty:
    st.success(f"Showing results for {selected_date}")
    st.dataframe(filtered.drop(columns=["only_date", "extracted_date"]))
else:
    st.error(f"No data found for {selected_date}")

# ✅ Optional: View all available dates
with st.expander("📅 Show all available dates in CSV"):
    st.write(sorted(df["only_date"].unique()))
