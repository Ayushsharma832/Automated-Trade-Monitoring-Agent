import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

st.set_page_config(page_title="Trade Anomaly Viewer", page_icon="📊", layout="wide")
st.title("📈 Historical Anomaly Dashboard")

if not os.path.exists("events_log.jsonl"):
    st.warning("No anomalies recorded yet.")
else:
    with open("events_log.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)

    if df.empty:
        st.info("✅ No anomalies found in logs.")
    else:
        # ✅ Convert timestamp to readable format
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.strftime("%d %b %Y, %I:%M %p")

        st.success(f"📊 Total anomalies recorded: {len(df)}")

        # Filter by symbol
        symbol_filter = st.selectbox("Filter by symbol", ["All"] + sorted(df["symbol"].unique().tolist()))
        if symbol_filter != "All":
            df = df[df["symbol"] == symbol_filter]

        # Display table sorted by latest
        st.dataframe(
            df[["timestamp", "symbol", "price", "anomaly_explanation"]]
            .sort_values("timestamp", ascending=False)
            .reset_index(drop=True)
        )
