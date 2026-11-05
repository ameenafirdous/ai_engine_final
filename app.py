import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# INDmoney SIP Discovery Engine
# --------------------------------------------------

st.set_page_config(
    page_title="INDmoney SIP Discovery Engine",
    page_icon="📊",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
CSV_CANDIDATES = [
    "final.csv",
    "finall.csv",
    "finall(2).csv",
    "final(2).csv",
    "data.csv",
]

PRESET_QUESTIONS = [
    "Why do first-time investors start SIPs, and what do they expect?",
    "What motivates investors to continue their SIPs?",
    "What are the reasons investors pause, reduce, or stop SIPs?",
    "What payment, autopay, or mandate issues affect SIP continuity?",
    "How do affordability and cash-flow constraints affect SIPs?",
    "What concerns do investors express about returns, losses, or risk?",
    "What app, support, or trust issues appear in the feedback?",
    "What confuses investors about SIP status, transactions, units, or NAV?",
    "What patterns appear between SIP continuers and drop-offs?",
]

OPPORTUNITY_MAP = {
    "SIP stopping / cancellation": (
        "Clarify SIP continuation and cancellation",
        "Help users understand their SIP status and next steps.",
    ),
    "SIP pause / restart": (
        "Make pause and restart easier",
        "Provide clear pause, restart, and status information.",
    ),
    "Failed SIP payment": (
        "Improve failed-payment recovery",
        "Explain payment failures and actionable next steps.",
    ),
    "Autopay / mandate": (
        "Improve mandate setup and troubleshooting",
        "Clarify mandate approval, activation, and payment permissions.",
    ),
    "Missed instalment": (
        "Support recovery after a missed instalment",
        "Show what happened and explain available next steps.",
    ),
    "SIP amount reduction": (
        "Support flexible contribution planning",
        "Make contribution changes easier to understand and manage.",
    ),
    "Insufficient balance / salary delay": (
        "Design for irregular cash flow",
        "Help users understand payment timing and available options.",
    ),
    "Emergency expenses": (
        "Support users during financial interruptions",
        "Clearly explain pause, modification, and restart options.",
    ),
    "Disappointment with returns": (
        "Set clearer long-term expectations",
        "Explain SIP mechanics and performance context without promises.",
    ),
    "Negative returns / losses": (
        "Improve risk and portfolio understanding",
        "Provide neutral explanations of fluctuations and risk.",
    ),
    "Confusion about SIP status": (
        "Make SIP status unmistakable",
        "Use consistent statuses and explain what each means.",
    ),
    "Pending transactions / unit allocation": (
        "Clarify transaction progress",
        "Explain processing stages and pending unit allocation.",
    ),
    "NAV / units / portfolio value": (
        "Explain investment values clearly",
        "Help users distinguish NAV, units, and portfolio value.",
    ),
    "Reasons for starting / expectations": (
        "Connect SIP setup to investor goals",
        "Help users understand their commitment and track progress.",
    ),
    "Reasons for continuing": (
        "Reinforce useful progress signals",
        "Show contribution history and goal progress without pressure.",
    ),
    "Reasons for discontinuing": (
        "Address avoidable continuation friction",
        "Prioritize issues that make continuing difficult.",
    ),
    "Fear / uncertainty": (
        "Reduce uncertainty with clear explanations",
        "Provide factual education and transparent next steps.",
    ),
    "Trust issues with app": (
        "Strengthen trust through transparency",
        "Make transactions, statuses, and support pathways verifiable.",
    ),
    "Customer support issues": (
        "Improve support resolution visibility",
        "Help users find support and understand case progress.",
    ),
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def find_column(df, candidates):
    normalized_columns = {normalize(col): col for col in df.columns}

    for candidate in candidates:
        key = normalize(candidate)
        if key in normalized_columns:
            return normalized_columns[key]

    for candidate in candidates:
        key = normalize(candidate)
        for normalized_col, original_col in normalized_columns.items():
            if key and (key in normalized_col or normalized_col in key):
                return original_col

    return None


@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path, encoding_errors="replace")


def locate_csv():
    for filename in CSV_CANDIDATES:
        path = ROOT / filename
        if path.exists():
            return path

    files = sorted(ROOT.glob("*.csv"))
    return files[0] if files else None


def prepare_data(raw):
    df = raw.copy()

    text_col = find_column(df, [
        "Comment Text", "comment_text", "review_text",
        "feedback_text", "review", "feedback", "text",
        "content", "message",
    ])
    source_col = find_column(df, ["Source", "platform", "source_name"])
    date_col = find_column(df, ["Date", "Date ", "review_date", "created_at"])
    url_col = find_column(df, [
        "Source URL (as collected)", "Source URL", "url", "link",
    ])
    label_col = find_column(df, [
        "Original Theme (unverified)", "Original Theme",
        "theme", "label", "labels", "category", "issue_label", "topic",
    ])
    id_col = find_column(df, ["Record ID", "feedback_id", "review_id", "id"])

    if text_col is None:
        raise ValueError(
            "Could not identify the feedback text column. "
            "Expected a column such as 'Comment Text' or 'review_text'."
        )

    df["_text"] = df[text_col].fillna("").astype(str).str.strip()
    df = df[df["_text"].ne("")].copy()

    df["_source"] = (
        df[source_col].fillna("Unknown").astype(str)
        if source_col else "Unknown"
    )
    df["_date"] = (
        pd.to_datetime(df[date_col], errors="coerce")
        if date_col else pd.NaT
    )
    df["_url"] = (
        df[url_col].fillna("").astype(str)
        if url_col else ""
    )
    df["_label"] = (
        df[label_col].fillna("Unlabelled").astype(str).str.strip()
        if label_col else "Unlabelled"
    )
    df["_id"] = (
        df[id_col].fillna("").astype(str)
        if id_col else df.index.astype(str)
    )

    df["_year"] = df["_date"].dt.year.astype("Int64").astype(str)
    df.loc[df["_date"].isna(), "_year"] = "Unknown"

    return df, {
        "text": text_col,
        "source": source_col,
        "date": date_col,
        "url": url_col,
        "label": label_col,
        "id": id_col,
    }


def tokenize(text):
    stop_words = {
        "the", "and", "for", "that", "this", "with", "from",
        "have", "has", "are", "was", "were", "but", "not",
        "you", "your", "they", "them", "their", "about",
        "into", "when", "what", "why", "how", "can", "could",
        "would", "should", "will", "just", "very", "also",
        "app", "indmoney", "sip", "sips", "investment",
        "invest", "investing",
    }
    words = re.findall(r"[a-z]{3,}", normalize(text))
    return {word for word in words if word not in stop_words}


def rank_evidence(df, question, limit=20):
    """Rank CSV rows by question terms, related concepts, and theme matches."""
    q = normalize(question)
    q_tokens = tokenize(question)

    concept_terms = {
        "start": [
            "started", "begin", "first", "expectation", "goal", "reason",
        ],
        "continue": [
            "continuation", "regular", "habit", "long term",
            "keep investing", "stay invested",
        ],
        "stop": [
            "stopped", "cancel", "cancellation", "discontinue",
            "drop off", "withdraw",
        ],
        "pause": [
            "paused", "restart", "resume", "skip", "missed instalment",
        ],
        "payment": [
            "failed", "failure", "mandate", "autopay",
            "auto debit", "bank", "debit",
        ],
        "afford": [
            "salary", "balance", "expense", "cash flow",
            "reduce amount", "financial difficulty",
        ],
        "risk": [
            "return", "loss", "negative", "market",
            "fear", "uncertain", "volatility",
        ],
        "app": [
            "support", "trust", "customer service",
            "complaint", "experience",
        ],
        "status": [
            "transaction", "pending", "nav", "unit",
            "portfolio", "allotment",
        ],
        "segment": [
            "income", "age", "first time investor", "drop off",
        ],
    }

    expanded_tokens = set(q_tokens)

    for concept, terms in concept_terms.items():
        if concept in q:
            for term in terms:
                expanded_tokens.update(tokenize(term))

    ranked = []

    for index, row in df.iterrows():
        text = row["_text"]
        text_norm = normalize(text)
        text_tokens = tokenize(text)
        label_tokens = tokenize(row["_label"])

        text_overlap = len(expanded_tokens & text_tokens)
        theme_overlap = len(q_tokens & label_tokens)

        phrase_bonus = sum(
            3 for term in q_tokens
            if len(term) > 3 and term in text_norm.split()
        )

        score = text_overlap + (3 * theme_overlap) + phrase_bonus

        # Boost rows with exact theme terms from the query.
        if normalize(row["_label"]) in q:
            score += 5

        if score > 0:
            ranked.append((score, index, row))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:limit]


def show_feedback(row):
    st.markdown(f"> {row['_text']}")

    metadata = [str(row["_source"])]

    if pd.notna(row["_date"]):
        metadata.append(row["_date"].strftime("%d %b %Y"))

    if row["_label"] and row["_label"] != "Unlabelled":
        metadata.append(f"Theme: {row['_label']}")

    st.caption(" · ".join(metadata))

    if row["_url"].strip():
        st.markdown(f"[Open original source]({row['_url']})")


def build_opportunities(df):
    counts = df["_label"].value_counts()
    rows = []

    for theme, count in counts.items():
        theme_norm = normalize(theme)
        mapping = None

        for key, value in OPPORTUNITY_MAP.items():
            key_norm = normalize(key)
            if key_norm in theme_norm or theme_norm in key_norm:
                mapping = value
                break

        if mapping is None:
            mapping = (
                f"Investigate: {theme}",
                "Review the original feedback to identify the underlying user problem.",
            )

        rows.append({
            "Theme": theme,
            "Feedback count": int(count),
            "Share": count / len(df),
            "Opportunity": mapping[0],
            "Potential direction": mapping[1],
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Load data
# --------------------------------------------------

st.title("INDmoney SIP Discovery Engine")
st.caption(
    "Explore user feedback to understand SIP continuation, "
    "drop-off, and product opportunities."
)

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a section",
        [
            "Dashboard",
            "Insights",
            "Opportunity Areas",
            "Pipeline",
            "Ask AI",
            "Evidence Browser",
        ],
        label_visibility="collapsed",
    )

csv_path = locate_csv()

if csv_path is None:
    st.error(
        "No CSV file found. Place your dataset beside app.py "
        "(for example, final.csv or finall.csv)."
    )
    st.stop()

try:
    raw_data = load_csv(str(csv_path))
    data, column_info = prepare_data(raw_data)
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

with st.sidebar:
    st.success(f"Loaded: {csv_path.name}")
    st.metric("Feedback records", f"{len(data):,}")


# --------------------------------------------------
# Dashboard filters ONLY
# --------------------------------------------------

if page == "Dashboard":
    with st.expander("Dashboard filters", expanded=False):
        sources = sorted(data["_source"].dropna().unique())
        themes = sorted(data["_label"].dropna().unique())

        selected_sources = st.multiselect(
            "Platform / source",
            sources,
            default=sources,
            key="dashboard_sources",
        )
        selected_themes = st.multiselect(
            "Theme",
            themes,
            default=themes,
            key="dashboard_themes",
        )

        filtered = data[
            data["_source"].isin(selected_sources)
            & data["_label"].isin(selected_themes)
        ].copy()
else:
    filtered = data.copy()

if filtered.empty:
    st.warning("No records match the selected Dashboard filters.")
    st.stop()


# --------------------------------------------------
# Dashboard
# --------------------------------------------------

if page == "Dashboard":
    st.subheader("Dataset overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Feedback records", f"{len(filtered):,}")
    c2.metric("Platforms", filtered["_source"].nunique())
    c3.metric("Themes", filtered["_label"].nunique())

    dates = filtered["_date"].dropna()
    if len(dates):
        c4.metric(
            "Date range",
            f"{dates.min():%b %Y} – {dates.max():%b %Y}",
        )
    else:
        c4.metric("Date range", "Unavailable")

    left, right = st.columns(2)

    with left:
        source_counts = (
            filtered["_source"]
            .value_counts()
            .rename_axis("Platform")
            .reset_index(name="Records")
        )
        st.plotly_chart(
            px.bar(
                source_counts,
                x="Platform",
                y="Records",
                title="Feedback by platform",
            ),
            use_container_width=True,
        )

    with right:
        theme_counts = (
            filtered["_label"]
            .value_counts()
            .head(12)
            .rename_axis("Theme")
            .reset_index(name="Records")
        )
        st.plotly_chart(
            px.bar(
                theme_counts,
                x="Records",
                y="Theme",
                orientation="h",
                title="Top feedback themes",
            ),
            use_container_width=True,
        )

    if filtered["_date"].notna().any():
        trend = filtered.dropna(subset=["_date"]).copy()
        trend["_month"] = trend["_date"].dt.to_period("M").astype(str)
        trend = trend.groupby("_month").size().reset_index(name="Records")

        st.plotly_chart(
            px.line(
                trend,
                x="_month",
                y="Records",
                markers=True,
                title="Feedback over time",
            ),
            use_container_width=True,
        )


# --------------------------------------------------
# Insights
# --------------------------------------------------

elif page == "Insights":
    st.subheader("Insights from collected feedback")
    st.caption(
        "Observations are calculated from the CSV. Keyword counts are "
        "directional and should be checked against original feedback."
    )

    total = len(data)
    theme_counts = data["_label"].value_counts()
    source_counts = data["_source"].value_counts()

    if len(theme_counts):
        theme = theme_counts.index[0]
        count = int(theme_counts.iloc[0])
        st.markdown(
            f"- The most common recorded theme is **{theme}** "
            f"({count:,} records; {count / total:.1%})."
        )

    if len(source_counts):
        source = source_counts.index[0]
        count = int(source_counts.iloc[0])
        st.markdown(
            f"- **{source}** is the largest feedback source "
            f"({count:,} records; {count / total:.1%})."
        )

    payment_terms = ["payment", "mandate", "autopay", "debit", "failed"]
    payment_count = data["_text"].str.lower().apply(
        lambda text: any(term in text for term in payment_terms)
    ).sum()

    st.markdown(
        f"- {payment_count:,} records contain payment or mandate-related "
        f"terms ({payment_count / total:.1%}). This is a keyword count, "
        f"not a manually validated classification."
    )

    theme_chart = theme_counts.head(12).reset_index()
    theme_chart.columns = ["Theme", "Records"]

    st.plotly_chart(
        px.pie(
            theme_chart,
            names="Theme",
            values="Records",
            hole=0.45,
            title="Feedback distribution by theme",
        ),
        use_container_width=True,
    )

    st.markdown("#### Feedback examples")
    for _, row in data.head(5).iterrows():
        with st.container(border=True):
            show_feedback(row)


# --------------------------------------------------
# Opportunity Areas
# --------------------------------------------------

elif page == "Opportunity Areas":
    st.subheader("Opportunity areas")
    st.caption(
        "Opportunity hypotheses are mapped from dataset themes. "
        "Feedback volume does not necessarily indicate severity or impact."
    )

    opportunities = build_opportunities(data).head(15)
    left, right = st.columns([1.2, 1])

    with left:
        st.plotly_chart(
            px.bar(
                opportunities.sort_values("Feedback count"),
                x="Feedback count",
                y="Theme",
                orientation="h",
                title="Feedback volume by theme",
            ),
            use_container_width=True,
        )

    with right:
        st.plotly_chart(
            px.treemap(
                opportunities,
                path=["Theme"],
                values="Feedback count",
                title="Opportunity landscape",
            ),
            use_container_width=True,
        )

    st.markdown("#### Opportunity hypotheses")

    for _, row in opportunities.iterrows():
        with st.container(border=True):
            st.markdown(f"**{row['Opportunity']}**")
            st.caption(
                f"{row['Theme']} · {row['Feedback count']:,} records "
                f"· {row['Share']:.1%} of feedback"
            )
            st.write(row["Potential direction"])

            examples = data[data["_label"] == row["Theme"]].head(2)
            for _, evidence in examples.iterrows():
                st.markdown(f"> {evidence['_text']}")
                if evidence["_url"].strip():
                    st.markdown(f"[Source]({evidence['_url']})")


# --------------------------------------------------
# Pipeline
# --------------------------------------------------

elif page == "Pipeline":
    st.subheader("Discovery pipeline")

    steps = [
        ("1. Collect", "Collect feedback and preserve text, source, date, and URL."),
        ("2. Load and validate", "Load the CSV, identify columns, and remove empty feedback."),
        ("3. Explore", "Review platform, theme, and date distributions."),
        ("4. Retrieve evidence", "Match research questions to feedback using text and theme relevance."),
        ("5. Synthesize", "Summarize evidence and formulate opportunity hypotheses."),
        ("6. Validate", "Review original quotes and source links before reporting findings."),
    ]

    for title, description in steps:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(description)

    st.markdown("#### Dataset checks")

    checks = [
        ("CSV loaded", csv_path.name),
        ("Feedback text column", column_info["text"]),
        ("Source column", column_info["source"] or "Not detected"),
        ("Date column", column_info["date"] or "Not detected"),
        ("Source URL column", column_info["url"] or "Not detected"),
        ("Theme column", column_info["label"] or "Not detected"),
    ]

    for label, value in checks:
        st.write(f"✓ **{label}:** {value}")


# --------------------------------------------------
# Ask AI
# --------------------------------------------------

elif page == "Ask AI":
    st.subheader("Ask a research question")
    st.caption(
        "Searches the loaded CSV and returns relevant feedback, theme summaries, "
        "and source links. Results are evidence retrieval, not causal conclusions."
    )

    def sync_preset():
        selected = st.session_state.get("ask_preset", "Choose a question…")
        if selected != "Choose a question…":
            st.session_state["ask_question"] = selected

    if "ask_question" not in st.session_state:
        st.session_state["ask_question"] = ""

    st.selectbox(
        "Research question presets",
        ["Choose a question…"] + PRESET_QUESTIONS,
        key="ask_preset",
        on_change=sync_preset,
    )

    question = st.text_area(
        "Your research question",
        height=100,
        placeholder="For example: What causes investors to stop or pause their SIP?",
        key="ask_question",
    )

    if st.button("Analyze CSV feedback", type="primary", key="ask_search"):
        if not question.strip():
            st.warning("Choose a preset or enter a question.")
        else:
            results = rank_evidence(data, question, limit=25)

            if not results:
                st.warning(
                    "No relevant records were found. Try broader wording "
                    "or browse the Evidence Browser by theme."
                )
            else:
                matched = pd.DataFrame([item[2] for item in results])
                theme_counts = matched["_label"].value_counts()
                source_counts = matched["_source"].value_counts()

                st.markdown("#### What the dataset suggests")
                st.write(
                    f"The search found **{len(results)} matching feedback records** "
                    f"out of **{len(data):,} loaded records**. This is qualitative "
                    f"retrieval, not a representative survey estimate."
                )

                if len(theme_counts):
                    st.markdown("**Most represented themes in retrieved evidence:**")
                    summary = (
                        theme_counts.head(5)
                        .rename_axis("Theme")
                        .reset_index(name="Matching records")
                    )
                    st.dataframe(
                        summary,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.plotly_chart(
                        px.bar(
                            summary.sort_values("Matching records"),
                            x="Matching records",
                            y="Theme",
                            orientation="h",
                            title="Themes among retrieved feedback",
                        ),
                        use_container_width=True,
                    )

                if len(source_counts):
                    st.caption(
                        "Sources represented: "
                        + ", ".join(
                            f"{source} ({count})"
                            for source, count in source_counts.items()
                        )
                    )

                st.markdown("#### Supporting user feedback")
                for score, index, row in results[:12]:
                    with st.container(border=True):
                        show_feedback(row)

                st.info(
                    "Review original quotes before using them as findings. "
                    "Keyword retrieval can miss relevant records or return false matches."
                )


# --------------------------------------------------
# Evidence Browser
# --------------------------------------------------

elif page == "Evidence Browser":
    st.subheader("Evidence browser")
    st.caption(
        "Browse feedback using dropdown filters for year, platform, and theme."
    )

    browser = data.copy()

    years = sorted(
        browser.loc[browser["_year"] != "Unknown", "_year"].unique(),
        reverse=True,
    )
    platforms = sorted(browser["_source"].dropna().unique())
    themes = sorted(browser["_label"].dropna().unique())

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_year = st.selectbox(
            "Year",
            ["All years"] + years,
            key="evidence_year",
        )

    with col2:
        selected_platform = st.selectbox(
            "Platform",
            ["All platforms"] + platforms,
            key="evidence_platform",
        )

    with col3:
        selected_theme = st.selectbox(
            "Theme / label",
            ["All themes"] + themes,
            key="evidence_theme",
        )

    if selected_year != "All years":
        browser = browser[browser["_year"] == selected_year]

    if selected_platform != "All platforms":
        browser = browser[browser["_source"] == selected_platform]

    if selected_theme != "All themes":
        browser = browser[browser["_label"] == selected_theme]

    search = st.text_input(
        "Optional text search",
        placeholder="Search within selected feedback...",
        key="evidence_search",
    )

    if search.strip():
        browser = browser[
            browser["_text"].str.contains(
                re.escape(search.strip()),
                case=False,
                na=False,
            )
        ]

    st.caption(
        f"Showing {min(len(browser), 100):,} of "
        f"{len(browser):,} matching records."
    )

    if browser.empty:
        st.info("No feedback matches these selections.")
    else:
        browser = browser.sort_values(
            "_date",
            ascending=False,
            na_position="last",
        )

        for _, row in browser.head(100).iterrows():
            with st.container(border=True):
                show_feedback(row)
