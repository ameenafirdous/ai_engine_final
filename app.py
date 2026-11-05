import re
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

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
    "Why do investors pause, reduce, or stop SIPs?",
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


def normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def find_column(df, candidates):
    columns = {normalize(col): col for col in df.columns}

    for candidate in candidates:
        key = normalize(candidate)
        if key in columns:
            return columns[key]

    for candidate in candidates:
        key = normalize(candidate)
        for normalized_col, original_col in columns.items():
            if key and (key in normalized_col or normalized_col in key):
                return original_col

    return None


@st.cache_data(show_spinner=False)
def read_csv(path):
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

    if not text_col:
        raise ValueError(
            "Feedback text column not found. Expected a column such as "
            "'Comment Text' or 'review_text'."
        )

    df["_text"] = df[text_col].fillna("").astype(str).str.strip()
    df = df[df["_text"] != ""].copy()

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


def retrieve_evidence(df, question, limit=12):
    question_norm = normalize(question)
    question_tokens = tokenize(question)

    expansions = {
        "start": ["started", "begin", "first sip", "expectation", "goal"],
        "continue": ["continuation", "regular", "habit", "long term", "stay invested"],
        "stop": ["stopped", "cancel", "cancellation", "discontinue", "drop off"],
        "pause": ["paused", "restart", "resume", "skip", "missed instalment"],
        "payment": ["failed", "failure", "mandate", "autopay", "auto debit", "bank"],
        "afford": ["salary", "balance", "expense", "cash flow", "reduce amount"],
        "risk": ["return", "loss", "negative", "market", "fear", "uncertain"],
        "app": ["support", "trust", "customer service", "complaint", "experience"],
        "status": ["transaction", "pending", "nav", "unit", "portfolio", "allotment"],
    }

    expanded_tokens = set(question_tokens)

    for key, terms in expansions.items():
        if key in question_norm:
            for term in terms:
                expanded_tokens.update(tokenize(term))

    results = []

    for index, row in df.iterrows():
        text = normalize(row["_text"])
        text_tokens = tokenize(row["_text"])
        theme_tokens = tokenize(row["_label"])

        text_overlap = len(expanded_tokens & text_tokens)
        theme_overlap = len(question_tokens & theme_tokens)
        phrase_bonus = sum(
            2 for term in expansions.get(question_norm, [])
            if normalize(term) in text
        )

        score = text_overlap + (2 * theme_overlap) + phrase_bonus

        if score > 0:
            results.append((score, index, row))

    results.sort(key=lambda item: item[0], reverse=True)
    return results[:limit]


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
        mapping = None
        theme_norm = normalize(theme)

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
# Load dataset
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
        "No CSV found. Place your dataset beside app.py "
        "(for example, final.csv or finall.csv)."
    )
    st.stop()

try:
    raw_data = read_csv(str(csv_path))
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

        dashboard_data = data[
            data["_source"].isin(selected_sources)
            & data["_label"].isin(selected_themes)
        ].copy()
else:
    dashboard_data = data.copy()

if dashboard_data.empty:
    st.warning("No records match the selected dashboard filters.")
    st.stop()


# --------------------------------------------------
# Dashboard
# --------------------------------------------------

if page == "Dashboard":
    st.subheader("Dataset overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Feedback records", f"{len(dashboard_data):,}")
    c2.metric("Platforms", dashboard_data["_source"].nunique())
    c3.metric("Themes", dashboard_data["_label"].nunique())

    dates = dashboard_data["_date"].dropna()
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
            dashboard_data["_source"]
            .value_counts()
            .rename_axis("Platform")
            .reset_index(name="Records")
        )
        st.plotly_chart(
            px.bar(source_counts, x="Platform", y="Records",
                   title="Feedback by platform"),
            use_container_width=True,
        )

    with right:
        theme_counts = (
            dashboard_data["_label"]
            .value_counts()
            .head(12)
            .rename_axis("Theme")
            .reset_index(name="Records")
        )
        st.plotly_chart(
            px.bar(theme_counts, x="Records", y="Theme",
                   orientation="h", title="Top feedback themes"),
            use_container_width=True,
        )

    if dashboard_data["_date"].notna().any():
        trend = dashboard_data.dropna(subset=["_date"]).copy()
        trend["_month"] = trend["_date"].dt.to_period("M").astype(str)
        trend = trend.groupby("_month").size().reset_index(name="Records")

        st.plotly_chart(
            px.line(trend, x="_month", y="Records", markers=True,
                    title="Feedback over time"),
            use_container_width=True,
        )


# --------------------------------------------------
# Insights
# --------------------------------------------------

elif page == "Insights":
    st.subheader("Insights from collected feedback")
    st.caption(
        "These are dataset-derived observations. Keyword matches are directional "
        "and should be checked against the original feedback."
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
        "Opportunity hypotheses are mapped from your dataset's themes. "
        "Feedback volume is not the same as severity or business impact."
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
# Ask AI / Evidence retrieval
# --------------------------------------------------

elif page == "Ask AI":
    st.subheader("Ask a research question")
    st.caption(
        "Searches the CSV and returns ranked evidence with source links. "
        "This is evidence retrieval, not an AI-generated causal conclusion."
    )

    if "ask_question" not in st.session_state:
        st.session_state["ask_question"] = ""

    def update_question():
        chosen = st.session_state.get("ask_preset", "")
        if chosen and chosen != "Choose a question…":
            st.session_state["ask_question"] = chosen

    st.selectbox(
        "Research question presets",
        ["Choose a question…"] + PRESET_QUESTIONS,
        key="ask_preset",
        on_change=update_question,
    )

    question = st.text_area(
        "Your research question",
        key="ask_question",
        height=100,
        placeholder="Enter a question about SIP starts, continuation, or drop-off...",
    )

    if st.button("Search evidence", type="primary"):
        if not question.strip():
            st.warning("Choose a preset or enter a research question.")
        else:
            results = retrieve_evidence(data, question, limit=12)

            if not results:
                st.warning(
                    "No matching evidence found. Try related wording or "
                    "browse the feedback by theme."
                )
            else:
                st.markdown("#### Evidence-based response")
                st.write(
                    f"Retrieved **{len(results)}** relevant records "
                    f"from **{len(data):,}** feedback records."
                )

                result_df = pd.DataFrame([item[2] for item in results])
                themes = result_df["_label"].value_counts()

                st.write(
                    "Themes represented: "
                    + ", ".join(
                        f"{theme} ({count})"
                        for theme, count in themes.head(5).items()
                    )
                )

                st.markdown("#### Supporting feedback")
                for score, index, row in results:
                    with st.container(border=True):
                        show_feedback(row)

                st.info(
                    "Treat these as qualitative signals. Review the original "
                    "quotes before using them as findings."
                )


# --------------------------------------------------
# Evidence Browser
# --------------------------------------------------

elif page == "Evidence Browser":
    st.subheader("Evidence browser")
    st.caption(
        "Browse feedback by platform, assigned theme, or month. "
        "These controls apply only to the Evidence Browser."
    )

    mode = st.radio(
        "Browse by",
        ["Platform", "Theme", "Month"],
        horizontal=True,
        key="evidence_mode",
    )

    browser = data.copy()

    if mode == "Platform":
        options = sorted(browser["_source"].dropna().unique())
        selected = st.multiselect(
            "Select platform(s)",
            options,
            default=options,
            key="evidence_platforms",
        )
        browser = browser[browser["_source"].isin(selected)]

    elif mode == "Theme":
        options = sorted(browser["_label"].dropna().unique())
        selected = st.multiselect(
            "Select theme(s)",
            options,
            default=options,
            key="evidence_themes",
        )
        browser = browser[browser["_label"].isin(selected)]

    else:
        dated = browser[browser["_date"].notna()].copy()
        dated["_month"] = dated["_date"].dt.to_period("M").astype(str)
        options = sorted(dated["_month"].unique(), reverse=True)

        selected = st.multiselect(
            "Select month(s)",
            options,
            default=options,
            key="evidence_months",
        )

        browser = dated[dated["_month"].isin(selected)]

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
