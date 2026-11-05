
import streamlit as st
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="INDmoney SIP Discovery Engine",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
DATA_PATH = Path(__file__).parent / "finall_labeled.csv"

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()

    data = pd.read_csv(DATA_PATH)

    # Clean column names without renaming original fields
    data.columns = data.columns.str.strip()

    # Convert review dates
    if "review_date" in data.columns:
        data["review_date"] = pd.to_datetime(
            data["review_date"],
            errors="coerce",
            dayfirst=True
        )

    return data

df = load_data()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("INDmoney SIP Discovery Engine")
st.caption(
    "AI-powered discovery from real user feedback"
)

if df.empty:
    st.error(
        "Dataset not found. Make sure finall_labeled.csv "
        "is in the same GitHub repository as app.py."
    )
    st.stop()

# --------------------------------------------------
# VALIDATE REQUIRED COLUMNS
# --------------------------------------------------
required_columns = [
    "feedback_id",
    "source_platform",
    "original_text",
    "review_date",
    "source_url",
    "labels"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("Discovery Engine")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Explore Feedback",
        "Evidence Browser"
    ]
)

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
if page == "Dashboard":

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Feedback",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Issue Categories",
            df["labels"].nunique()
        )

    with col3:
        st.metric(
            "Feedback Sources",
            df["source_platform"].nunique()
        )

    st.divider()

    # Category distribution
    st.subheader("Feedback by Issue Category")

    label_counts = (
        df["labels"]
        .fillna("Uncategorized")
        .value_counts()
    )

    st.bar_chart(label_counts)

    # Source distribution
    st.subheader("Feedback by Source")

    source_counts = (
        df["source_platform"]
        .fillna("Unknown")
        .value_counts()
    )

    st.bar_chart(source_counts)

    # Category summary table
    st.subheader("Category Summary")

    summary = (
        df.groupby("labels", dropna=False)
        .agg(
            Feedback_Count=("feedback_id", "count"),
            Sources=("source_platform", "nunique")
        )
        .reset_index()
        .rename(columns={"labels": "Issue Category"})
        .sort_values(
            "Feedback_Count",
            ascending=False
        )
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# EXPLORE FEEDBACK
# --------------------------------------------------
elif page == "Explore Feedback":

    st.header("Explore User Feedback")

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        categories = sorted(
            df["labels"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_category = st.selectbox(
            "Filter by issue category",
            ["All categories"] + categories
        )

    with col2:
        sources = sorted(
            df["source_platform"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_source = st.selectbox(
            "Filter by source",
            ["All sources"] + sources
        )

    search_query = st.text_input(
        "Search feedback",
        placeholder="Search keywords in user feedback..."
    )

    filtered_df = df.copy()

    if selected_category != "All categories":
        filtered_df = filtered_df[
            filtered_df["labels"] == selected_category
        ]

    if selected_source != "All sources":
        filtered_df = filtered_df[
            filtered_df["source_platform"] == selected_source
        ]

    if search_query.strip():
        filtered_df = filtered_df[
            filtered_df["original_text"]
            .astype(str)
            .str.contains(
                search_query.strip(),
                case=False,
                na=False,
                regex=False
            )
        ]

    st.caption(
        f"Showing {len(filtered_df):,} of {len(df):,} feedback records"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    # Download filtered results
    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download filtered feedback",
        data=csv_data,
        file_name="filtered_sip_feedback.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# EVIDENCE BROWSER
# --------------------------------------------------
elif page == "Evidence Browser":

    st.header("Evidence Browser")

    st.write(
        "Review individual feedback records, their assigned "
        "issue labels, and their original source links."
    )

    selected_id = st.selectbox(
        "Select a feedback record",
        df["feedback_id"].astype(str).tolist()
    )

    record = df[
        df["feedback_id"].astype(str) == selected_id
    ].iloc[0]

    st.subheader(f"Feedback ID: {record['feedback_id']}")

    st.markdown("**Issue Label**")
    st.write(record["labels"])

    st.markdown("**Source**")
    st.write(record["source_platform"])

    st.markdown("**Review Date**")
    st.write(
        record["review_date"].strftime("%d %b %Y")
        if pd.notna(record["review_date"])
        else "Unknown"
    )

    st.markdown("**Original Feedback**")
    st.info(str(record["original_text"]))

    if pd.notna(record["source_url"]):
        st.markdown(
            f"[Open original source]({record['source_url']})"
        )

    st.divider()

    st.subheader("Dataset Quality")

    quality_col1, quality_col2, quality_col3 = st.columns(3)

    with quality_col1:
        st.metric(
            "Missing Feedback Text",
            int(df["original_text"].isna().sum())
        )

    with quality_col2:
        st.metric(
            "Missing Labels",
            int(df["labels"].isna().sum())
        )

    with quality_col3:
        st.metric(
            "Duplicate Feedback IDs",
            int(df["feedback_id"].duplicated().sum())
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Research tool for product discovery. "
    "Feedback labels are exploratory classifications, "
    "not investment advice."
)
