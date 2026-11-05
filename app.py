import re
import os
import json
import urllib.request
import urllib.error
from pathlib import Path
from collections import Counter

import pandas as pd
import streamlit as st
import plotly.express as px

# ------------------------------------------------------------
# INDmoney SIP Discovery Engine
# Local CSV-first Streamlit app. No API key required.
# ------------------------------------------------------------

st.set_page_config(
    page_title="INDmoney SIP Discovery Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Evidence-led discovery for improving first-time SIP continuation."},
)

ROOT = Path(__file__).resolve().parent
CSV_CANDIDATES = [
    ROOT / "final.csv",
    ROOT / "finall.csv",
    ROOT / "finall(2).csv",
    ROOT / "final(2).csv",
    ROOT / "data.csv",
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
    "What differences or patterns appear between SIP continuers and drop-offs?",
]

THEME_OPPORTUNITIES = {
    "SIP stopping / cancellation": (
        "Make SIP continuation and cancellation states clearer",
        "Help users understand the consequences and next steps before they stop an SIP.",
    ),
    "SIP pause / restart": (
        "Make pause and restart easier to understand",
        "Provide clear, reliable controls and status visibility for pausing or restarting.",
    ),
    "Failed SIP payment": (
        "Recover failed instalments with actionable guidance",
        "Explain why an instalment failed and what the user can do next.",
    ),
    "Autopay / mandate": (
        "Improve mandate setup and troubleshooting",
        "Reduce uncertainty around mandate approval, activation, and payment permissions.",
    ),
    "Missed instalment": (
        "Help users recover after a missed instalment",
        "Show the missed payment clearly and explain available next steps without pressure.",
    ),
    "SIP amount reduction": (
        "Support flexible contribution planning",
        "Make it easier to understand and manage contribution changes.",
    ),
    "Insufficient balance / salary delay": (
        "Design for irregular cash flow",
        "Help users anticipate payment timing and manage SIP commitments around income cycles.",
    ),
    "Emergency expenses": (
        "Acknowledge financial interruptions",
        "Offer clear information about pause, modification, and restart options during disruptions.",
    ),
    "Disappointment with returns": (
        "Set clearer long-term expectations",
        "Explain performance context and SIP mechanics without promising returns.",
    ),
    "Negative returns / losses": (
        "Improve risk and portfolio understanding",
        "Help users interpret fluctuations with neutral, educational explanations.",
    ),
    "Confusion about SIP status": (
        "Make SIP status unmistakable",
        "Use consistent statuses and explain what each status means.",
    ),
    "Pending transactions / unit allocation": (
        "Clarify transaction progress",
        "Show expected processing stages and explain pending unit allocation.",
    ),
    "NAV / units / portfolio value": (
        "Explain investment values in plain language",
        "Make NAV, units, and portfolio value easier to distinguish.",
    ),
    "Reasons for starting / expectations": (
        "Connect SIP setup to the investor's stated goal",
        "Help users understand their commitment and track progress toward their own goal.",
    ),
    "Reasons for continuing": (
        "Reinforce useful progress signals",
        "Show factual contribution history and goal progress without urgency or fear.",
    ),
    "Reasons for discontinuing": (
        "Identify and address avoidable friction",
        "Use feedback to prioritize issues that make continuation difficult.",
    ),
    "Fear / uncertainty": (
        "Reduce uncertainty with clear explanations",
        "Provide calm, factual education and transparent next steps.",
    ),
    "Trust issues with app": (
        "Strengthen trust through transparency",
        "Make transactions, statuses, and support pathways easy to verify.",
    ),
    "Customer support issues": (
        "Improve support resolution visibility",
        "Help users find relevant support and understand case progress.",
    ),
}

QUESTION_HINTS = {
    "start": ["start", "begin", "first", "expect", "goal", "reason"],
    "continue": ["continue", "continu", "motivat", "habit", "long term", "progress"],
    "stop": ["stop", "stopp", "cancel", "discontinu", "withdraw", "exit"],
    "pause": ["pause", "restart", "resume", "tempor"],
    "payment": ["payment", "failed", "mandate", "autopay", "bank", "debit"],
    "afford": ["afford", "salary", "balance", "expense", "cash", "amount", "reduce"],
    "risk": ["return", "loss", "negative", "risk", "market", "fear"],
    "app": ["app", "support", "trust", "customer service", "complaint"],
    "clarity": ["status", "transaction", "pending", "nav", "unit", "portfolio"],
    "segment": ["continue", "drop", "stop", "income", "age", "segment"],
}


def norm(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def find_column(df, candidates):
    lookup = {norm(c): c for c in df.columns}
    for candidate in candidates:
        if norm(candidate) in lookup:
            return lookup[norm(candidate)]
    # Fuzzy fallback for column names with suffixes / punctuation.
    for candidate in candidates:
        c_norm = norm(candidate)
        for actual_norm, actual in lookup.items():
            if c_norm and (c_norm in actual_norm or actual_norm in c_norm):
                return actual
    return None


@st.cache_data(show_spinner=False)
def load_csv(path_string):
    return pd.read_csv(path_string, encoding_errors="replace")


def discover_csv():
    for path in CSV_CANDIDATES:
        if path.exists():
            return path
    csvs = sorted(ROOT.glob("*.csv"))
    return csvs[0] if csvs else None


def prepare_data(raw):
    df = raw.copy()
    text_col = find_column(df, [
        "Comment Text", "Original Text", "Review Text", "Feedback Text",
        "comment_text", "review_text", "feedback_text", "reviewText",
        "review", "feedback", "text", "content", "message",
    ])
    source_col = find_column(df, ["Source", "platform", "source_name"])
    date_col = find_column(df, ["Date", "Date ", "review_date", "created_at"])
    url_col = find_column(df, ["Source URL (as collected)", "Source URL", "url", "link"])
    label_col = find_column(df, [
        "Original Theme (unverified)", "Original Theme", "theme",
        "label", "labels", "category", "issue_label", "topic",
    ])
    id_col = find_column(df, ["Record ID", "feedback_id", "review_id", "id"])

    if text_col is None:
        raise ValueError(
            "Could not identify the feedback text column. Expected a column such as "
            "'Comment Text' or 'review_text'."
        )

    df["_text"] = df[text_col].fillna("").astype(str).str.strip()
    df = df[df["_text"].ne("")].copy()
    df["_source"] = df[source_col].fillna("Unknown").astype(str) if source_col else "Unknown"
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    df["_url"] = df[url_col].fillna("").astype(str) if url_col else ""
    df["_label"] = df[label_col].fillna("Unlabelled").astype(str).str.strip() if label_col else "Unlabelled"
    df["_id"] = df[id_col].fillna("").astype(str) if id_col else df.index.astype(str)
    return df, {
        "text": text_col, "source": source_col, "date": date_col,
        "url": url_col, "label": label_col, "id": id_col,
    }


def tokens(text):
    stop = {
        "the", "and", "for", "that", "this", "with", "from", "have", "has",
        "are", "was", "were", "but", "not", "you", "your", "they", "them",
        "their", "about", "into", "when", "what", "why", "how", "can", "could",
        "would", "should", "will", "just", "very", "also", "app", "indmoney",
        "sip", "sips", "investment", "invest", "investing", "does", "did",
        "users", "user", "people", "much", "many", "more", "less",
    }
    return {t for t in re.findall(r"[a-z]{2,}", norm(text)) if t not in stop}


QUERY_CONCEPTS = {
    "start": ["started", "begin", "first", "goal", "expectation", "new investor"],
    "continue": ["continuing", "continued", "regular", "habit", "long term", "stay invested", "keep investing", "discipline", "goal"],
    "motivat": ["motivation", "motivated", "continue", "goal", "habit", "returns", "wealth", "future", "discipline"],
    "stop": ["stopped", "cancel", "cancellation", "discontinue", "drop", "withdraw", "quit"],
    "pause": ["paused", "restart", "resume", "skip", "missed instalment", "temporary"],
    "payment": ["failed", "failure", "mandate", "autopay", "auto debit", "bank", "debit", "payment"],
    "afford": ["salary", "balance", "expense", "cash flow", "reduce amount", "financial difficulty", "emergency"],
    "risk": ["return", "loss", "negative", "market", "fear", "uncertain", "volatility"],
    "app": ["support", "trust", "customer service", "complaint", "experience", "issue", "problem"],
    "status": ["transaction", "pending", "nav", "unit", "portfolio", "allotment", "status"],
    "confus": ["unclear", "understand", "explain", "confusion", "status", "details"],
    "drop": ["stopped", "cancel", "discontinue", "drop off", "missed", "pause"],
}


def rank_evidence(df, question, limit=40):
    """Rank records using direct question terms, synonym expansion, and labels."""
    q = norm(question)
    q_tokens = tokens(question)
    expanded = set(q_tokens)
    concepts = []

    for concept, terms in QUERY_CONCEPTS.items():
        if concept in q:
            concepts.append(concept)
            for term in terms:
                expanded.update(tokens(term))

    if any(term in q for term in ["motivat", "continu", "stay invested"]):
        for concept in ["continue", "motivat"]:
            concepts.append(concept)
            expanded.update(tokens(" ".join(QUERY_CONCEPTS[concept])))

    ranked = []
    for idx, row in df.iterrows():
        text = str(row["_text"])
        label = str(row["_label"])
        text_norm, label_norm = norm(text), norm(label)
        text_tokens, label_tokens = tokens(text), tokens(label)

        direct = len(q_tokens & text_tokens)
        expanded_overlap = len(expanded & text_tokens)
        label_direct = len(q_tokens & label_tokens)
        label_expanded = len(expanded & label_tokens)

        score = 4 * direct + 2 * expanded_overlap + 7 * label_direct + 4 * label_expanded

        for concept in set(concepts):
            terms = QUERY_CONCEPTS.get(concept, [])
            if any(norm(term) in label_norm for term in terms):
                score += 8
            if any(norm(term) in text_norm for term in terms):
                score += 2

        if q and q in text_norm:
            score += 8

        # Avoid matches caused only by generic expanded terms.
        if direct == 0 and label_direct == 0 and expanded_overlap < 2 and label_expanded == 0:
            score = 0

        if score > 0:
            ranked.append((score, idx, row, direct, label_direct + label_expanded))

    ranked.sort(key=lambda x: (-x[0], -x[4], -x[3], str(x[2].get("_id", ""))))
    return ranked[:limit]


def summarize_results(results, total_records):
    if not results:
        return None
    matched = pd.DataFrame([item[2] for item in results])
    n = len(matched)

    patterns = {
        "Continuation / motivation": ["continue", "continuing", "long term", "goal", "future", "wealth", "regular", "discipline", "habit"],
        "Pause / stop / cancellation": ["stop", "stopped", "cancel", "pause", "paused", "restart", "discontinue", "withdraw"],
        "Payment / mandate": ["payment", "failed", "mandate", "autopay", "debit", "bank"],
        "Affordability / cash flow": ["salary", "balance", "expense", "emergency", "afford", "money", "cash", "amount"],
        "Returns / risk": ["return", "loss", "negative", "market", "risk", "profit"],
        "App / support / trust": ["app", "support", "customer care", "trust", "complaint", "issue"],
    }
    rows = []
    for name, terms in patterns.items():
        count = matched["_text"].str.lower().apply(
            lambda text: any(term in text for term in terms)
        ).sum()
        if count:
            rows.append({"Pattern": name, "Records": int(count), "Share of retrieved": count / n})

    return {
        "matched": matched,
        "count": n,
        "total": int(total_records),
        "themes": matched["_label"].value_counts(),
        "sources": matched["_source"].value_counts(),
        "patterns": pd.DataFrame(rows).sort_values("Records", ascending=False) if rows else pd.DataFrame(),
    }


def derive_insights(df):
    insights = []
    n = len(df)
    label_counts = df["_label"].value_counts()
    if len(label_counts):
        top_label, top_count = label_counts.index[0], int(label_counts.iloc[0])
        insights.append(
            f"The most frequent recorded theme is **{top_label}** "
            f"({top_count:,} of {n:,} feedback records; {top_count / n:.1%})."
        )
    source_counts = df["_source"].value_counts()
    if len(source_counts):
        top_source, top_count = source_counts.index[0], int(source_counts.iloc[0])
        insights.append(
            f"**{top_source}** is the largest source in the loaded dataset "
            f"({top_count:,} records; {top_count / n:.1%})."
        )
    payment_terms = ["payment", "mandate", "autopay", "debit", "failed"]
    payment_count = df["_text"].str.lower().apply(
        lambda t: any(term in t for term in payment_terms)
    ).sum()
    if payment_count:
        insights.append(
            f"{payment_count:,} records mention payment or mandate-related terms "
            f"({payment_count / n:.1%}); these are keyword matches, not manually validated classifications."
        )
    continuity_terms = ["stop", "stopped", "cancel", "pause", "missed", "continue", "restart"]
    continuity_count = df["_text"].str.lower().apply(
        lambda t: any(term in t for term in continuity_terms)
    ).sum()
    if continuity_count:
        insights.append(
            f"{continuity_count:,} records contain SIP-continuity terms "
            f"({continuity_count / n:.1%}); terms can occur in different contexts."
        )
    return insights


def opportunity_rows(df):
    counts = df["_label"].value_counts()
    rows = []
    for label, count in counts.items():
        label_norm = norm(label)
        mapping = None
        for key, value in THEME_OPPORTUNITIES.items():
            if norm(key) == label_norm or norm(key) in label_norm or label_norm in norm(key):
                mapping = value
                break
        if mapping is None:
            mapping = (
                f"Investigate: {label}",
                f"Review the underlying feedback to determine what user problem, if any, this theme represents.",
            )
        rows.append({
            "Theme": label,
            "Feedback count": int(count),
            "Share": float(count / len(df)),
            "Opportunity hypothesis": mapping[0],
            "Potential direction": mapping[1],
        })
    return pd.DataFrame(rows)


def show_evidence(row):
    st.markdown("**Original user feedback**")
    st.markdown(f"> {row['_text']}")
    metadata = [str(row["_source"])]
    if pd.notna(row["_date"]):
        metadata.append(row["_date"].strftime("%d %b %Y"))
    if str(row["_label"]).strip() and row["_label"] != "Unlabelled":
        metadata.append(f"Label: {row['_label']}")
    if str(row["_id"]).strip():
        metadata.append(f"Record: {row['_id']}")
    st.caption(" · ".join(metadata))
    url = str(row["_url"]).strip()
    if url and url.lower() not in {"nan", "none"}:
        st.markdown(f"[Open original source]({url})")


# ------------------------------------------------------------
# Visual identity: SIP-inspired navy/blue + gold/yellow
# ------------------------------------------------------------
if "app_theme" not in st.session_state:
    st.session_state.app_theme = "Light"

theme_col, _ = st.columns([1, 5])
with theme_col:
    st.session_state.app_theme = st.selectbox(
        "Appearance",
        ["Light", "Dark"],
        index=0 if st.session_state.app_theme == "Light" else 1,
        key="appearance_selector",
        label_visibility="collapsed",
    )

if st.session_state.app_theme == "Dark":
    bg = "#0B1220"
    panel = "#111C2E"
    text = "#E8EEF8"
    muted = "#A8B6CC"
    border = "#26364D"
    chart_template = "plotly_dark"
else:
    bg = "#F7F9FC"
    panel = "#FFFFFF"
    text = "#172B4D"
    muted = "#5B6B82"
    border = "#DCE5F0"
    chart_template = "plotly_white"

st.markdown(
    f"""
    <style>
      :root {{
        --sip-blue: #174A8B;
        --sip-blue-bright: #2563EB;
        --sip-gold: #F4C542;
        --sip-yellow: #FFE27A;
        --text-color: {text};
      }}
      .stApp {{ background: {bg}; color: {text}; }}
      [data-testid="stSidebar"] {{ background: {panel}; border-right: 1px solid {border}; }}
      [data-testid="stSidebar"] * {{ color: {text}; }}
      h1, h2, h3 {{ color: {text}; }}
      .sip-hero {{
        background: linear-gradient(115deg, #123B70 0%, #1D5AA6 62%, #F4C542 150%);
        color: white; border-radius: 18px; padding: 1.25rem 1.5rem;
        margin: .2rem 0 1rem 0; border: 1px solid rgba(244,197,66,.35);
      }}
      .sip-hero h1 {{ color: white !important; margin: 0 0 .25rem 0; font-size: 1.8rem; }}
      .sip-hero p {{ color: #EAF2FF; margin: 0; }}
      div[data-testid="stMetric"] {{
        background: {panel}; border: 1px solid {border}; padding: .8rem 1rem;
        border-radius: 14px;
      }}
      div[data-testid="stMetricLabel"] {{ color: {muted}; }}
      .stButton button, .stDownloadButton button {{
        border-radius: 10px; border: 1px solid #D9AE24;
      }}
      .stButton button[kind="primary"], .stDownloadButton button {{
        background: #F4C542; color: #17345C; font-weight: 700;
      }}
      [data-testid="stExpander"] {{ border-color: {border}; border-radius: 12px; }}
      .sip-empty {{ color: {muted}; text-align: center; padding: 1.5rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sip-hero">
      <h1>💙 INDmoney SIP Discovery Engine <span style="color:#FFE27A">₹</span></h1>
      <p>Discover evidence-backed opportunities to help first-time investors stay consistent with their SIPs.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🪙 Navigation")
    page = st.radio(
        "Choose a section",
        [
            "Dashboard",
            "Failure Explorer",
            "Pattern Explorer",
            "Candidate Problem Areas",
            "Evidence Explorer",
            "Ask the Evidence",
            "Insights",
            "Opportunity Areas",
            "Pipeline",
            "Quality & Integrity",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("💰 Data source: local CSV file beside app.py.")


csv_path = discover_csv()
if csv_path is None:
    st.error("No CSV file found. Place your dataset beside app.py (for example, final.csv or finall.csv).")
    st.stop()

try:
    raw_df = load_csv(str(csv_path))
    data, columns = prepare_data(raw_df)
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

with st.sidebar:
    st.success(f"Loaded: {csv_path.name}")
    st.download_button(
        "⬇️ Download dataset CSV",
        data=raw_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="indmoney_sip_feedback.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.metric("Feedback records", f"{len(data):,}")
    if columns["label"]:
        st.caption(f"Theme column: {columns['label']}")
    else:
        st.warning("No theme/label column detected. Opportunity analysis will use 'Unlabelled'.")

px.defaults.template = chart_template

# Filters are intentionally available only on the Dashboard.
if page == "Dashboard":
    with st.expander("Dashboard filters", expanded=False):
        sources = sorted(data["_source"].dropna().unique().tolist())
        selected_sources = st.multiselect("Platform / source", sources, default=sources, key="dashboard_sources")
        labels = sorted(data["_label"].dropna().unique().tolist())
        selected_labels = st.multiselect("Theme", labels, default=labels, key="dashboard_themes")
        filtered = data[
            data["_source"].isin(selected_sources) &
            data["_label"].isin(selected_labels)
        ].copy()
else:
    filtered = data.copy()

if filtered.empty:
    st.warning("No records match the selected dashboard filters.")
    st.stop()

if page == "Dashboard":
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Feedback records", f"{len(filtered):,}")
    c2.metric("Sources", f"{filtered['_source'].nunique():,}")
    c3.metric("Themes", f"{filtered['_label'].nunique():,}")
    date_values = filtered["_date"].dropna()
    c4.metric("Date range", f"{date_values.min():%b %Y} – {date_values.max():%b %Y}" if len(date_values) else "Not available")

    left, right = st.columns(2)
    with left:
        source_counts = filtered["_source"].value_counts().rename_axis("Source").reset_index(name="Records")
        st.plotly_chart(px.bar(source_counts, x="Source", y="Records", title="Feedback by source"), use_container_width=True)
    with right:
        theme_counts = filtered["_label"].value_counts().head(12).rename_axis("Theme").reset_index(name="Records")
        st.plotly_chart(px.bar(theme_counts, x="Records", y="Theme", orientation="h", title="Top feedback themes"), use_container_width=True)

    if filtered["_date"].notna().any():
        trend = filtered.dropna(subset=["_date"]).assign(Month=lambda x: x["_date"].dt.to_period("M").astype(str))
        trend = trend.groupby("Month").size().reset_index(name="Records")
        st.plotly_chart(px.line(trend, x="Month", y="Records", markers=True, title="Feedback over time"), use_container_width=True)

elif page == "Insights":
    st.subheader("Insights from the collected feedback")
    st.caption("These observations are calculated from the currently filtered CSV. Keyword counts are directional and should be validated against the original quotes.")
    for insight in derive_insights(filtered):
        st.markdown(f"- {insight}")

    st.markdown("#### Theme distribution")
    counts = filtered["_label"].value_counts().reset_index()
    counts.columns = ["Theme", "Records"]
    counts["Share"] = counts["Records"] / len(filtered)
    st.plotly_chart(
        px.pie(counts.head(12), names="Theme", values="Records", hole=0.45, title="Share of feedback by theme (top 12)"),
        use_container_width=True,
    )

    st.markdown("#### Evidence examples")
    for _, row in filtered.head(5).iterrows():
        with st.container(border=True):
            show_evidence(row)

elif page == "Opportunity Areas":
    st.subheader("Opportunity areas")
    st.caption("Opportunity hypotheses are mapped from feedback themes. Counts show the evidence volume, not the severity or business impact.")
    opportunities = opportunity_rows(filtered)
    top = opportunities.head(12).copy()

    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(
            px.bar(top.sort_values("Feedback count"), x="Feedback count", y="Theme", orientation="h",
                   title="Evidence volume by opportunity theme"),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            px.treemap(top, path=["Theme"], values="Feedback count", title="Opportunity landscape"),
            use_container_width=True,
        )

    st.markdown("#### Opportunity hypotheses")
    for _, row in top.iterrows():
        with st.container(border=True):
            st.markdown(f"**{row['Opportunity hypothesis']}**")
            st.caption(f"{row['Theme']} · {row['Feedback count']:,} records · {row['Share']:.1%} of filtered feedback")
            st.write(row["Potential direction"])
            matches = filtered[filtered["_label"] == row["Theme"]].head(2)
            for _, evidence in matches.iterrows():
                st.markdown(f"> {evidence['_text']}")
                if evidence["_url"].strip():
                    st.markdown(f"[Source]({evidence['_url']})")

elif page == "Pipeline":
    st.subheader("Discovery pipeline")
    st.caption(
        "A transparent, CSV-first workflow that moves from collected feedback to "
        "evidence-backed opportunity hypotheses. Each stage shows what the app actually does."
    )

    # Compact visual styling scoped to this page's pipeline cards.
    st.markdown(
        """
        <style>
        .pipeline-kicker {font-size: .76rem; letter-spacing: .09em; text-transform: uppercase;
            color: #64748b; font-weight: 700; margin-bottom: .35rem;}
        .pipeline-stage {border: 1px solid rgba(100,116,139,.24); border-radius: 14px;
            padding: 16px 16px 14px; min-height: 178px; background: rgba(148,163,184,.045);}
        .pipeline-number {display: inline-flex; align-items: center; justify-content: center;
            width: 30px; height: 30px; border-radius: 9px; background: #e2e8f0;
            color: #0f172a; font-weight: 800; margin-bottom: 10px;}
        .pipeline-stage h4 {font-size: 1rem; margin: 0 0 .45rem 0;}
        .pipeline-stage p {font-size: .88rem; line-height: 1.45; margin: 0; color: var(--text-color);}
        .pipeline-flow {text-align:center; color:#94a3b8; font-size:1.2rem; padding:2px 0 8px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    stages = [
        ("Collect", "Gather public feedback and retain the original text and available source metadata.", "Input"),
        ("Load & validate", "Load the local CSV, detect columns, remove blank feedback, and normalize fields for analysis.", "Data preparation"),
        ("Explore", "Review record volume, sources, themes, and dates. Dashboard filters apply to the Dashboard only.", "Explore"),
        ("Retrieve evidence", "Match the research question against feedback text and theme labels using keywords and synonym expansion.", "Search"),
        ("Synthesize", "Calculate counts for retrieved records, show themes and sources, and map themes to opportunity hypotheses.", "Evidence summary"),
        ("Validate", "Inspect verbatim quotes and source links, manually review classifications, and record limitations in the Evidence Log.", "Human review"),
    ]

    st.markdown("<div class='pipeline-kicker'>From raw feedback to validated learning</div>", unsafe_allow_html=True)
    for row_start in range(0, len(stages), 3):
        cols = st.columns(3, gap="medium")
        for col, (name, description, kind), idx in zip(
            cols, stages[row_start:row_start + 3], range(row_start + 1, row_start + 4)
        ):
            with col:
                st.markdown(
                    f"<div class='pipeline-stage'>"
                    f"<div class='pipeline-number'>{idx:02d}</div>"
                    f"<h4>{name}</h4>"
                    f"<p>{description}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        if row_start == 0:
            st.markdown("<div class='pipeline-flow'>↓ &nbsp; ↓ &nbsp; ↓</div>", unsafe_allow_html=True)

    st.markdown("### Dataset readiness")
    st.caption("Checks reflect the currently loaded CSV and detected columns—not a claim that the feedback is fully validated.")
    checks = [
        ("CSV loaded", True, csv_path.name),
        ("Feedback text detected", bool(columns["text"]), str(columns["text"] or "Not detected")),
        ("Source/platform detected", bool(columns["source"]), str(columns["source"] or "Not available")),
        ("Date detected", bool(columns["date"]), str(columns["date"] or "Not available")),
        ("Source URL detected", bool(columns["url"]), str(columns["url"] or "Not available")),
        ("Theme/label detected", bool(columns["label"]), str(columns["label"] or "Not available")),
    ]
    ready_count = sum(1 for _, passed, _ in checks if passed)
    metric_col, table_col = st.columns([1, 2.6], gap="large")
    with metric_col:
        st.metric("Metadata checks passed", f"{ready_count}/{len(checks)}")
        st.progress(ready_count / len(checks))
        st.caption("A detected field may still contain missing or inaccurate values.")
    with table_col:
        check_df = pd.DataFrame(
            [{"Check": label, "Status": "Available" if passed else "Missing", "Details": detail}
             for label, passed, detail in checks]
        )
        st.dataframe(check_df, use_container_width=True, hide_index=True)

    with st.expander("Method and interpretation notes"):
        st.markdown("""
        - **Retrieval:** keyword overlap, synonym expansion, and available theme labels.
        - **Counts:** describe records retrieved from this dataset; they are not population estimates.
        - **Opportunity areas:** hypotheses mapped from feedback themes, not proven causes or measured impact.
        - **Validation:** check the original quote and source, manually label a sample, and document disagreements.
        - **Business metric:** the target is the share of first-time mutual-fund investors with at least one successful SIP instalment in the 30 days around month 12. This feedback workflow alone cannot measure that outcome or establish causality.
        """)

elif page == "Ask AI":
    st.subheader("Ask a research question")
    st.caption(
        "Evidence-led analysis of the loaded CSV: patterns, directional counts, "
        "opportunity hypotheses, original feedback, and relevance to the 12-month SIP metric."
    )

    def sync_preset_to_question():
        selected = st.session_state.get("ask_preset", "Choose a question…")
        if selected != "Choose a question…":
            st.session_state["ask_question"] = selected

    if "ask_question" not in st.session_state:
        st.session_state["ask_question"] = ""

    st.selectbox(
        "Research question presets",
        ["Choose a question…"] + PRESET_QUESTIONS,
        key="ask_preset",
        on_change=sync_preset_to_question,
    )
    question = st.text_area(
        "Your question",
        height=100,
        placeholder="For example: What causes investors to stop or pause their SIP?",
        key="ask_question",
    )

    if st.button("Analyze CSV feedback", type="primary", key="ask_search"):
        if not question.strip():
            st.warning("Choose a preset or enter a question.")
        else:
            results = rank_evidence(data, question, limit=40)
            if not results:
                st.warning(
                    "No relevant rows were retrieved from this CSV. Try broader wording "
                    "or browse the Evidence Browser by theme."
                )
            else:
                matched = pd.DataFrame([item[2] for item in results])
                n_retrieved = len(matched)
                theme_counts = matched["_label"].value_counts()
                source_counts = matched["_source"].value_counts()

                st.markdown("### 1. Brief answer")
                st.write(
                    f"The search retrieved **{n_retrieved} relevant records** from "
                    f"**{len(data):,} loaded feedback records**. The evidence below "
                    "shows recurring feedback related to your question. Retrieval is "
                    "keyword/theme-based, so this is not a representative survey estimate."
                )

                st.markdown("### 2. Key patterns")
                pattern_defs = {
                    "SIP continuation / motivation": ["continue", "continuing", "long term", "goal", "future", "wealth", "regular", "discipline", "habit", "stay invested"],
                    "Pause / stop / cancellation": ["stop", "stopped", "cancel", "pause", "paused", "restart", "discontinue", "withdraw", "missed"],
                    "Payment / mandate friction": ["payment", "failed", "mandate", "autopay", "debit", "bank", "instalment", "installment"],
                    "Affordability / cash flow": ["salary", "balance", "expense", "emergency", "afford", "cash", "amount", "income", "reduce"],
                    "Returns / risk / uncertainty": ["return", "loss", "negative", "market", "risk", "profit", "volatility", "uncertain"],
                    "App / support / trust": ["app", "support", "customer care", "trust", "complaint", "issue", "problem"],
                    "Status / transaction clarity": ["status", "transaction", "pending", "nav", "unit", "portfolio", "allotment"],
                }
                pattern_rows = []
                for pattern_name, terms in pattern_defs.items():
                    mask = matched["_text"].str.lower().apply(
                        lambda text: any(term in text for term in terms)
                    )
                    count = int(mask.sum())
                    if count:
                        pattern_rows.append({
                            "Pattern": pattern_name,
                            "Retrieved records": count,
                            "Share of retrieved": count / n_retrieved,
                        })
                if pattern_rows:
                    pattern_df = pd.DataFrame(pattern_rows).sort_values(
                        "Retrieved records", ascending=False
                    )
                    st.dataframe(
                        pattern_df.style.format({"Share of retrieved": "{:.1%}"}),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "Patterns can overlap. Counts are automated keyword matches within "
                        "the retrieved records, not manually validated classifications."
                    )
                else:
                    st.write("No predefined pattern was detected; inspect the original evidence below.")

                st.markdown("### 3. Quantification and source coverage")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Retrieved themes**")
                    theme_summary = theme_counts.head(8).rename_axis("Theme").reset_index(name="Records")
                    theme_summary["Share of retrieved"] = theme_summary["Records"] / n_retrieved
                    st.dataframe(
                        theme_summary.style.format({"Share of retrieved": "{:.1%}"}),
                        use_container_width=True,
                        hide_index=True,
                    )
                with col_b:
                    st.markdown("**Sources represented**")
                    source_summary = source_counts.rename_axis("Source").reset_index(name="Records")
                    st.dataframe(source_summary, use_container_width=True, hide_index=True)

                st.markdown("### 4. Opportunity areas to investigate")
                st.caption(
                    "Ranked by retrieved evidence volume only. Volume is not severity, "
                    "causal impact, or proof that an intervention will improve continuation."
                )
                opportunity_rows_local = []
                for theme, count in theme_counts.items():
                    label_norm = norm(theme)
                    mapped = None
                    for key, value in THEME_OPPORTUNITIES.items():
                        if norm(key) == label_norm or norm(key) in label_norm or label_norm in norm(key):
                            mapped = value
                            break
                    if mapped is None:
                        mapped = (
                            f"Investigate: {theme}",
                            "Review the original feedback to determine whether this theme represents an actionable user problem."
                        )
                    opportunity_rows_local.append({
                        "Theme": theme,
                        "Retrieved records": int(count),
                        "Opportunity hypothesis": mapped[0],
                        "Potential direction": mapped[1],
                    })
                opportunity_df = pd.DataFrame(opportunity_rows_local).head(6)
                for _, opp in opportunity_df.iterrows():
                    with st.container(border=True):
                        st.markdown(f"**{opp['Opportunity hypothesis']}**")
                        st.caption(f"{opp['Theme']} · {opp['Retrieved records']} retrieved records")
                        st.write(opp["Potential direction"])

                st.markdown("### 5. Original user evidence")
                st.caption("Quotes are displayed verbatim from the CSV; metadata and source links are retained.")
                for item in results[:12]:
                    score, idx, row, direct_score, label_score = item
                    with st.container(border=True):
                        show_evidence(row)

                st.markdown("### 6. Connection to the business metric")
                st.write(
                    "These findings may indicate friction or motivations that could influence "
                    "whether a first-time investor has at least one successful SIP instalment "
                    "in the 30 days around the 12-month mark. The feedback dataset does not "
                    "by itself establish that a theme causes SIP drop-off or changes the "
                    "12-month active-SIP continuation rate. Treat each opportunity as a hypothesis "
                    "to validate with user research and product/behavioral data."
                )

                st.markdown("### 7. Evidence quality and limitations")
                st.write(
                    f"- **Evidence base:** {n_retrieved} retrieved records from {len(data):,} loaded records.\n"
                    "- **Method:** keyword overlap, synonym expansion, and available theme labels; no LLM-generated facts.\n"
                    "- **Quantification:** automated keyword counts within retrieved records; patterns may overlap and need manual review.\n"
                    "- **Representativeness:** collected public feedback may be self-selected and does not represent all investors.\n"
                    "- **Causality:** quotes and themes suggest hypotheses, not causal explanations or measured impact.\n"
                    "- **Validation:** inspect the original quote and source link, then hand-label a sample and record agreement in the AI Evidence Log."
                )

elif page == "Failure Explorer":
    st.subheader("Failure explorer")
    st.caption("Explore potential SIP journey failures using transparent keyword screens.")
    failure_defs = {
        "Payment / debit failure": ["failed payment", "payment failed", "debit failed", "transaction failed", "payment failure"],
        "Mandate / autopay": ["mandate", "autopay", "auto debit", "e mandate", "e-mandate"],
        "Missed instalment": ["missed instalment", "missed installment", "missed sip", "instalment failed", "installment failed"],
        "Pending transaction / units": ["pending", "unit allocation", "allotment", "processing"],
        "App / support issue": ["customer support", "customer care", "not working", "app issue", "complaint"],
        "Pause / cancellation": ["pause", "paused", "cancel", "cancellation", "stop sip", "stopped sip"],
    }
    options = list(failure_defs)
    selected = st.selectbox("Failure category", options)
    terms = failure_defs[selected]
    subset = data[data["_text"].str.lower().apply(lambda t: any(term in t for term in terms))]
    c1, c2 = st.columns(2)
    c1.metric("Keyword-matched records", f"{len(subset):,}")
    c2.metric("Share of dataset", f"{len(subset)/len(data):.1%}" if len(data) else "—")
    st.caption("Screening count only; keyword matches can include false positives and are not validated failure rates.")
    for _, row in subset.head(50).iterrows():
        with st.container(border=True): show_evidence(row)

elif page == "Pattern Explorer":
    st.subheader("Pattern explorer")
    st.caption("Compare recorded labels and directional keyword patterns across sources.")
    selected_pattern = st.selectbox("Pattern", [
        "Payment / mandate", "Affordability / cash flow", "Returns / risk",
        "Pause / stop", "Status / transaction clarity", "App / support / trust"
    ])
    terms_by_pattern = {
        "Payment / mandate": ["payment", "mandate", "autopay", "debit", "failed"],
        "Affordability / cash flow": ["salary", "balance", "expense", "emergency", "cash", "income", "reduce"],
        "Returns / risk": ["return", "loss", "negative", "market", "risk", "volatility"],
        "Pause / stop": ["pause", "paused", "stop", "stopped", "cancel", "discontinue"],
        "Status / transaction clarity": ["status", "pending", "transaction", "nav", "unit", "allotment"],
        "App / support / trust": ["app", "support", "customer care", "trust", "complaint", "issue"],
    }
    terms = terms_by_pattern[selected_pattern]
    mask = data["_text"].str.lower().apply(lambda t: any(term in t for term in terms))
    subset = data[mask].copy()
    st.metric("Keyword-matched records", f"{len(subset):,}", f"{len(subset)/len(data):.1%} of dataset" if len(data) else None)
    by_source = subset["_source"].value_counts().rename_axis("Source").reset_index(name="Records")
    if not by_source.empty:
        st.plotly_chart(px.bar(by_source, x="Source", y="Records", title=f"{selected_pattern} by source"), use_container_width=True)
    by_label = subset["_label"].value_counts().rename_axis("Theme").reset_index(name="Records")
    st.dataframe(by_label, use_container_width=True, hide_index=True)
    for _, row in subset.head(20).iterrows():
        with st.container(border=True): show_evidence(row)

elif page == "Candidate Problem Areas":
    st.subheader("Candidate problem areas")
    st.caption("Prioritization is an initial screening based on evidence volume and metric relevance—not a validated impact estimate.")
    opp = opportunity_rows(data)
    continuity_terms = ["sip", "continue", "stop", "pause", "cancel", "payment", "mandate", "missed", "instalment", "installment"]
    opp["Metric relevance screen"] = opp["Theme"].astype(str).str.lower().apply(
        lambda t: "Potentially relevant" if any(term in t for term in continuity_terms) else "Needs review"
    )
    opp["Evidence volume score"] = opp["Feedback count"] / max(1, opp["Feedback count"].max())
    opp["Screening score"] = opp["Evidence volume score"] * opp["Metric relevance screen"].map(
        {"Potentially relevant": 1.0, "Needs review": 0.5}
    )
    opp = opp.sort_values(["Screening score", "Feedback count"], ascending=False)
    st.dataframe(opp[["Theme", "Feedback count", "Share", "Metric relevance screen",
                      "Screening score", "Opportunity hypothesis"]].style.format(
                          {"Share": "{:.1%}", "Screening score": "{:.2f}"}),
                 use_container_width=True, hide_index=True)
    st.caption("Validate relevance by reviewing quotes and checking whether the issue plausibly affects SIP continuity.")

elif page == "Evidence Explorer":
    st.subheader("Evidence explorer")
    st.caption("Search original records and preserve traceability.")
    c1, c2, c3 = st.columns(3)
    sources = ["All"] + sorted(data["_source"].dropna().unique().tolist())
    labels = ["All"] + sorted(data["_label"].dropna().unique().tolist())
    years = ["All"] + sorted(data["_date"].dropna().dt.year.unique().tolist(), reverse=True)
    with c1: esource = st.selectbox("Source", sources, key="evidence_explorer_source")
    with c2: elabel = st.selectbox("Theme", labels, key="evidence_explorer_label")
    with c3: eyear = st.selectbox("Year", years, key="evidence_explorer_year")
    search = st.text_input("Search original text", key="evidence_explorer_search")
    view = data.copy()
    if esource != "All": view = view[view["_source"] == esource]
    if elabel != "All": view = view[view["_label"] == elabel]
    if eyear != "All": view = view[view["_date"].dt.year == int(eyear)]
    if search.strip(): view = view[view["_text"].str.contains(re.escape(search), case=False, na=False)]
    st.caption(f"{len(view):,} matching records")
    for _, row in view.sort_values("_date", ascending=False, na_position="last").head(100).iterrows():
        with st.container(border=True): show_evidence(row)

elif page == "Ask the Evidence":
    st.subheader("Ask the evidence")
    st.caption("Evidence-grounded synthesis. Configure an OpenAI-compatible API using environment variables or Streamlit secrets; otherwise the app displays a local evidence summary.")
    question = st.text_area("Research question", key="ask_evidence_question", placeholder="What might make first-time investors stop or pause their SIP?")
    if st.button("Analyze evidence", type="primary", key="ask_evidence_button"):
        results = rank_evidence(data, question, limit=30) if question.strip() else []
        if not results:
            st.warning("Enter a question and ensure the CSV contains relevant feedback.")
        else:
            retrieved = [item[2] for item in results]
            st.markdown("### Evidence retrieved")
            st.write(f"{len(retrieved)} records retrieved from {len(data)} total records.")
            for row in retrieved[:10]:
                with st.container(border=True): show_evidence(row)
            api_key = os.getenv("LLM_API_KEY", "")
            base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1/chat/completions")
            model = os.getenv("LLM_MODEL", "gpt-4o-mini")
            try:
                if "LLM_API_KEY" in st.secrets: api_key = st.secrets["LLM_API_KEY"]
                if "LLM_BASE_URL" in st.secrets: base_url = st.secrets["LLM_BASE_URL"]
                if "LLM_MODEL" in st.secrets: model = st.secrets["LLM_MODEL"]
            except Exception:
                pass
            if api_key:
                evidence_payload = []
                for i, row in enumerate(retrieved, 1):
                    evidence_payload.append({
                        "evidence_id": f"E{i}", "text": row["_text"],
                        "source": row["_source"], "date": str(row["_date"]),
                        "label": row["_label"], "record_id": row["_id"], "url": row["_url"]
                    })
                prompt = (
                    "You are a rigorous product discovery research analyst. Answer only from supplied CSV evidence. "
                    "Never invent quotes, counts, causal claims, or missing attributes. Separate observations, "
                    "interpretations, and hypotheses. Counts in evidence are retrieved-record counts, not population estimates. "
                    "Cite evidence IDs (E1 etc.) for every substantive claim. Discuss counter-evidence and limitations. "
                    "Connect cautiously to the business metric: first-time investors with >=1 successful SIP instalment "
                    "in the 30 days around month 12. Do not recommend specific funds or use fear/FOMO. "
                    f"Research question: {question}\\nEvidence JSON: {json.dumps(evidence_payload, ensure_ascii=False)}"
                )
                payload = json.dumps({
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Be evidence-grounded, concise, and explicit about uncertainty."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2
                }).encode("utf-8")
                req = urllib.request.Request(
                    base_url, data=payload,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                    method="POST"
                )
                try:
                    with urllib.request.urlopen(req, timeout=60) as response:
                        result = json.loads(response.read().decode("utf-8"))
                    answer = result["choices"][0]["message"]["content"]
                    st.markdown("### AI synthesis")
                    st.markdown(answer)
                except Exception as exc:
                    st.error(f"API request failed: {exc}. Showing retrieved evidence only.")
            else:
                st.info("No API key configured. Showing evidence retrieval only; no generative AI answer was produced.")
            st.markdown("### Evidence coverage")
            theme_summary = pd.DataFrame([r[2] for r in results])["_label"].value_counts().rename_axis("Theme").reset_index(name="Retrieved records")
            st.dataframe(theme_summary, use_container_width=True, hide_index=True)

elif page == "Quality & Integrity":
    st.subheader("Quality & integrity")
    n = len(data)
    quality_rows = [
        ("Non-empty feedback", n, n),
        ("Source present", int(data["_source"].astype(str).str.strip().ne("").sum()), n),
        ("Date present", int(data["_date"].notna().sum()), n),
        ("Source URL present", int(data["_url"].astype(str).str.strip().ne("").sum()), n),
        ("Theme/label present", int(data["_label"].astype(str).str.strip().ne("Unlabelled").sum()), n),
        ("Record ID present", int(data["_id"].astype(str).str.strip().ne("").sum()), n),
    ]
    quality = pd.DataFrame([{"Check": name, "Records": count, "Total": total,
                            "Coverage": count/total if total else 0}
                           for name, count, total in quality_rows])
    st.dataframe(quality.style.format({"Coverage": "{:.1%}"}), use_container_width=True, hide_index=True)
    st.markdown("### Integrity rules")
    st.markdown("- Original feedback text is displayed without rewriting.\\n- Keyword matches are directional and may overlap.\\n- Retrieved feedback is self-selected and not necessarily representative.\\n- Theme labels may be unverified; inspect source text.\\n- No causal effect or 12-month retention rate is calculated from this dataset.\\n- API-generated synthesis, when configured, is grounded in retrieved excerpts but still requires human verification.")

