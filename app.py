import re
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
        "Comment Text", "comment_text", "review_text", "feedback_text",
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
        "sip", "sips", "investment", "invest", "investing",
    }
    return {t for t in re.findall(r"[a-z]{3,}", norm(text).replace(" ", " ")) if t not in stop}


def rank_evidence(df, question, limit=8):
    q_tokens = tokens(question)
    q_norm = norm(question)
    ranked = []
    for idx, row in df.iterrows():
        text = row["_text"]
        text_tokens = tokens(text)
        overlap = len(q_tokens & text_tokens)
        label_tokens = tokens(row["_label"])
        overlap += 2 * len(q_tokens & label_tokens)
        # Small phrase boost for a question's salient words.
        phrase_boost = sum(1 for t in q_tokens if t in norm(text).split())
        score = overlap + 0.15 * phrase_boost
        if score > 0:
            ranked.append((score, idx, row))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[:limit]


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
    st.markdown(f"> {row['_text']}")
    meta = [str(row["_source"])]
    if pd.notna(row["_date"]):
        meta.append(row["_date"].strftime("%d %b %Y"))
    if row["_label"] and row["_label"] != "Unlabelled":
        meta.append(f"Theme: {row['_label']}")
    st.caption(" · ".join(meta))
    if row["_url"].strip():
        st.markdown(f"[Open source]({row['_url']})")


st.title("INDmoney SIP Discovery Engine")
st.caption("Evidence-led exploration of user feedback to understand SIP continuation and drop-off.")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a section",
        ["Dashboard", "Insights", "Opportunity Areas", "Pipeline", "Ask AI", "Evidence Browser"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Data source: local CSV file in the same folder as app.py.")

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
    st.metric("Feedback records", f"{len(data):,}")
    if columns["label"]:
        st.caption(f"Theme column: {columns['label']}")
    else:
        st.warning("No theme/label column detected. Opportunity analysis will use 'Unlabelled'.")

# Global filters
with st.expander("Filters", expanded=False):
    sources = sorted(data["_source"].dropna().unique().tolist())
    selected_sources = st.multiselect("Source", sources, default=sources)
    labels = sorted(data["_label"].dropna().unique().tolist())
    selected_labels = st.multiselect("Theme", labels, default=labels)
    filtered = data[
        data["_source"].isin(selected_sources) &
        data["_label"].isin(selected_labels)
    ].copy()

if filtered.empty:
    st.warning("No records match the selected filters.")
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
    st.caption("This is the workflow represented by the current CSV-first app.")
    steps = [
        ("1. Collect", "Gather public feedback and preserve source, date, text, and URL where available."),
        ("2. Load & validate", "Read the local CSV, identify the feedback-text column, remove empty text rows, and normalize metadata for analysis."),
        ("3. Filter & explore", "Filter by source and theme; inspect dataset volume, distributions, and time trends."),
        ("4. Retrieve evidence", "Match a research question to relevant feedback using keyword overlap and theme text."),
        ("5. Synthesize", "Use retrieved quotes and counts to formulate evidence-backed observations and opportunity hypotheses."),
        ("6. Validate", "Open source links, review original quotes, and label any interpretation as a hypothesis until validated."),
    ]
    for title, description in steps:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(description)

    st.markdown("#### Current data checks")
    checks = [
        ("Dataset loaded", True, f"{csv_path.name}"),
        ("Feedback text column detected", bool(columns["text"]), str(columns["text"])),
        ("Source metadata detected", bool(columns["source"]), str(columns["source"] or "Not available")),
        ("Date metadata detected", bool(columns["date"]), str(columns["date"] or "Not available")),
        ("Source URL detected", bool(columns["url"]), str(columns["url"] or "Not available")),
        ("Theme/label detected", bool(columns["label"]), str(columns["label"] or "Not available")),
    ]
    for label, passed, detail in checks:
        st.write(("✓" if passed else "—") + f" **{label}:** {detail}")

elif page == "Ask AI":
    st.subheader("Ask a research question")
    st.caption("Choose a research question or enter your own. Results are retrieved from the filtered CSV; no unsupported facts are generated.")
    selected_preset = st.selectbox("Research question presets", ["Select a question…"] + PRESET_QUESTIONS)
    question_default = selected_preset if selected_preset != "Select a question…" else ""
    question = st.text_area("Your question", value=question_default, height=90, placeholder="Ask about SIP starts, continuation, pauses, payment issues, or drop-off…")
    run = st.button("Find evidence", type="primary")

    if run:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            ranked = rank_evidence(filtered, question, limit=10)
            if not ranked:
                st.info("No matching evidence was found in the selected records. Try different wording or broaden the filters.")
            else:
                st.markdown("#### Retrieved evidence")
                st.caption(f"{len(ranked)} relevant records shown, ranked by text/theme keyword overlap. This is retrieval, not a causal analysis.")
                for score, idx, row in ranked:
                    with st.container(border=True):
                        show_evidence(row)
                st.markdown("#### Evidence summary")
                matched_df = pd.DataFrame([r[2] for r in ranked])
                top_themes = matched_df["_label"].value_counts().head(5)
                if len(top_themes):
                    st.write("Most represented themes in retrieved evidence:")
                    st.write(", ".join(f"{theme} ({count})" for theme, count in top_themes.items()))
                st.info("Review the quotes before using them as a finding. Keyword retrieval may miss relevant feedback or include false matches.")

elif page == "Evidence Browser":
    st.subheader("Evidence browser")
    search = st.text_input("Search feedback text", placeholder="Search words or phrases…")
    browser = filtered
    if search.strip():
        browser = browser[browser["_text"].str.contains(re.escape(search.strip()), case=False, na=False)]
    st.caption(f"{len(browser):,} records")
    for _, row in browser.head(100).iterrows():
        with st.container(border=True):
            show_evidence(row)
