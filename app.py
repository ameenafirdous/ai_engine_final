
import streamlit as st
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="INDmoney SIP Discovery Engine",
    page_icon="📊",
    layout="wide"
)

# Load dataset
DATA_PATH = Path(__file__).parent / "finall_labeled.csv"

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)

df = load_data()

# Header
st.title("INDmoney SIP Discovery Engine")
st.caption("AI-powered discovery from real user feedback")

if df.empty:
    st.error("Dataset not found. Please check that finall_labeled.csv is in the repository.")
    st.stop()

# Overview metrics
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Feedback", len(df))

with col2:
    label_column = "labels"
    st.metric(
        "Issue Categories",
        df[label_column].nunique() if label_column in df.columns else 0
    )

with col3:
    source_column = "Source"
    st.metric(
        "Feedback Sources",
        df[source_column].nunique() if source_column in df.columns else 0
    )

# Label distribution
st.subheader("Feedback by Issue Category")

if "labels" in df.columns:
    label_counts = df["labels"].value_counts()
    st.bar_chart(label_counts)

# Browse feedback
st.subheader("Explore User Feedback")

if "labels" in df.columns:
    selected_label = st.selectbox(
        "Filter by issue category",
        ["All categories"] + sorted(df["labels"].dropna().unique().tolist())
    )

    filtered_df = df.copy()

    if selected_label != "All categories":
        filtered_df = filtered_df[
            filtered_df["labels"] == selected_label
        ]

    search = st.text_input("Search feedback")

    if search:
        text_columns = filtered_df.select_dtypes(
            include=["object"]
        ).columns

        mask = filtered_df[text_columns].astype(str).apply(
            lambda col: col.str.contains(
                search, case=False, na=False
            )
        ).any(axis=1)

        filtered_df = filtered_df[mask]

    st.dataframe(filtered_df, use_container_width=True)
    st.caption(f"Showing {len(filtered_df)} feedback records")

