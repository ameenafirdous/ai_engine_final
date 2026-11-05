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
    "Why do people start a SIP, and what do they expect to happen?",
    "What changes in their behaviour in the months after starting?",
    "What leads investors to pause, reduce or stop a SIP?",
    "What do investors check, and how often, once they have started?",
    "What confuses or worries them inside the app?",
    "What do they do outside the app when they feel unsure?",
    "What separates investors who continue from those who quietly drop off?",
    "How do these patterns differ across user segments (income stability, age, source of money, who influenced them)?",
    "What unmet needs emerge consistently across conversations?",
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


def question_profile(question):
    """Return a narrow retrieval profile inferred from the research question."""
    q = norm(question)
    profiles = [
        ("start_expectations", ["start", "started", "begin", "expect", "expectation", "why do people start"]),
        ("behaviour_change", ["behaviour", "behavior", "months after", "after starting", "change in"]),
        ("pause_reduce_stop", ["pause", "reduce", "stop", "stopping", "cancel", "discontinu"]),
        ("monitoring", ["check", "checks", "how often", "monitor", "track", "look at"]),
        ("in_app_confusion", ["confus", "worr", "inside the app", "app", "status", "transaction"]),
        ("outside_app", ["outside the app", "when they feel unsure", "ask others", "youtube", "reddit", "advisor"]),
        ("continuers_dropoffs", ["continue", "drop off", "drop-off", "quietly drop", "continuers"]),
        ("segments", ["user segments", "income stability", "age", "source of money", "influenced them"]),
        ("unmet_needs", ["unmet needs", "needs emerge", "across conversations"]),
        ("affordability", ["afford", "salary", "cash flow", "cash-flow", "insufficient balance", "emergency"]),
        ("payments", ["payment", "mandate", "autopay", "failed instalment", "failed installment"]),
        ("returns_risk", ["return", "loss", "market", "risk", "volatility"]),
    ]
    for profile, phrases in profiles:
        if any(phrase in q for phrase in phrases):
            return profile
    return "custom"


PROFILE_ANCHORS = {
    "start_expectations": ["started sip", "start sip", "started investing", "first sip", "goal", "retirement", "wealth", "future", "expect", "expectation", "beginner", "recommend", "suggested"],
    "behaviour_change": ["after starting", "started checking", "check daily", "check frequently", "increase", "decrease", "changed", "habit", "regular", "monthly", "missed", "paused", "stopped", "continue", "track", "monitor"],
    "pause_reduce_stop": ["pause", "paused", "stop sip", "stopped sip", "cancel sip", "cancelled sip", "canceled sip", "reduce sip", "reduced amount", "discontinue", "missed instalment", "missed installment", "not enough balance", "salary", "emergency", "expense"],
    "monitoring": ["check", "checking", "track", "tracking", "portfolio", "returns", "nav", "units", "sip status", "transaction", "daily", "weekly", "monthly", "frequently", "every day"],
    "in_app_confusion": ["confus", "unclear", "not understand", "don't understand", "cannot understand", "status", "pending", "mandate", "nav", "units", "portfolio value", "error", "issue", "problem", "support"],
    "outside_app": ["youtube", "reddit", "google", "search online", "friend", "friends", "family", "advisor", "adviser", "financial planner", "whatsapp", "telegram", "ask someone", "outside the app"],
    "continuers_dropoffs": ["continue sip", "continued sip", "stopped sip", "stop sip", "cancel sip", "regular sip", "missed instalment", "missed installment", "discipline", "goal", "salary", "income", "returns", "loss", "emergency"],
    "segments": ["salary", "income", "age", "young", "retire", "parent", "family", "bonus", "business income", "irregular income", "influenced", "friend", "family", "advisor", "first salary", "student"],
    "unmet_needs": ["wish", "need", "would like", "should have", "missing", "feature", "help", "support", "confus", "unclear", "unable", "cannot", "could not", "frustrat", "problem", "issue"],
    "affordability": ["salary", "payday", "income", "insufficient balance", "low balance", "not enough balance", "cash flow", "cashflow", "afford", "affordable", "emergency", "medical", "unexpected expense", "financial problem", "reduce sip", "reduce amount", "missed instalment", "missed installment"],
    "payments": ["payment failed", "failed payment", "autopay", "auto debit", "mandate", "bank debit", "debit failed", "instalment failed", "installment failed", "payment pending"],
    "returns_risk": ["negative return", "loss", "market fall", "market crash", "volatility", "risk", "fear", "uncertain", "down market", "returns", "profit"],
}


def rank_evidence(df, question, limit=40):
    """Precision-first retrieval: require question-specific anchors, then rank matches."""
    q = norm(question)
    profile = question_profile(question)
    anchors = PROFILE_ANCHORS.get(profile, [])
    q_tokens = tokens(question)
    ranked = []

    for idx, row in df.iterrows():
        text = str(row["_text"])
        label = str(row["_label"])
        text_norm, label_norm = norm(text), norm(label)
        text_tokens, label_tokens = tokens(text), tokens(label)
        matched_anchors = [a for a in anchors if norm(a) in text_norm]

        if profile != "custom":
            # For known questions, incidental overlap with generic terms or labels
            # is not enough: require at least one specific anchor in the original text.
            if not matched_anchors:
                continue
            score = sum(5 + min(len(a.split()), 3) for a in matched_anchors)
            score += 2 * len(q_tokens & text_tokens)
            if any(norm(a) in label_norm for a in anchors):
                score += 1  # labels are a weak tie-breaker, never retrieval evidence
        else:
            expanded = set(q_tokens)
            for concept, terms in QUERY_CONCEPTS.items():
                if concept in q:
                    expanded.update(tokens(" ".join(terms)))
            direct = len(q_tokens & text_tokens)
            overlap = len(expanded & text_tokens)
            if direct == 0 and overlap < 2:
                continue
            score = 4 * direct + 2 * overlap
            matched_anchors = [t for t in sorted(q_tokens) if t in text_tokens]

        if score > 0:
            ranked.append((score, idx, row, len(matched_anchors), len(q_tokens & text_tokens)))

    ranked.sort(key=lambda x: (-x[0], -x[3], -x[4], str(x[2].get("_id", ""))))

    # Keep evidence relevant, but avoid allowing one platform to dominate the
    # displayed sample when other sources also contain strong matches.
    # Round-robin source selection preserves ranking within each source.
    by_source = {}
    for item in ranked:
        source = str(item[2].get("_source", "Unknown"))
        by_source.setdefault(source, []).append(item)
    selected = []
    while len(selected) < limit and any(by_source.values()):
        for source in list(by_source):
            if by_source[source] and len(selected) < limit:
                selected.append(by_source[source].pop(0))
    return selected


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


st.title("INDmoney SIP Discovery Engine")
st.caption("Evidence-led exploration of user feedback to understand SIP continuation and drop-off.")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a section",
        [
            "Research Overview",
            "Dashboard",
            "Failure Explorer",
            "Pattern Explorer",
            "Candidate Problem Areas",
            "Evidence Explorer",
            "Ask the Evidence",
            "Opportunity Areas",
            "Pipeline",
            "Quality & Integrity",
        ],
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
    # Editorial dashboard: all metrics are computed from the loaded CSV.
    st.markdown("""
    <style>
    .dash-hero {padding: 1.35rem 1.5rem; border-radius: 18px;
        background: linear-gradient(115deg, #102a43 0%, #174b62 58%, #247b78 100%);
        color: #ffffff; margin: .25rem 0 1.15rem 0;}
    .dash-eyebrow {font-size: .74rem; letter-spacing: .14em; text-transform: uppercase;
        opacity: .78; font-weight: 700; margin-bottom: .45rem;}
    .dash-hero h2 {color: #ffffff; font-size: 1.8rem; line-height: 1.15; margin: 0 0 .45rem 0;}
    .dash-hero p {color: #e5f3f4; font-size: .96rem; margin: 0; max-width: 850px;}
    .dash-kpi {border: 1px solid rgba(100,116,139,.22); border-radius: 15px;
        padding: 1rem 1.1rem; min-height: 118px; background: rgba(148,163,184,.045);}
    .dash-kpi-label {font-size: .78rem; color: #64748b; font-weight: 700; letter-spacing: .04em;}
    .dash-kpi-value {font-size: 1.85rem; font-weight: 800; line-height: 1.25; margin: .35rem 0; color: var(--text-color);}
    .dash-kpi-note {font-size: .78rem; color: #64748b;}
    .dash-section {font-size: 1.15rem; font-weight: 750; margin: 1.15rem 0 .25rem 0;}
    .dash-feature {border: 1px solid rgba(20,184,166,.35); border-radius: 15px;
        padding: 1rem 1.15rem; background: rgba(20,184,166,.06);}
    .dash-feature-title {font-weight: 750; font-size: 1.05rem; margin-bottom: .3rem;}
    .dash-feature-meta {font-size: .85rem; color: #64748b;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="dash-hero">
      <div class="dash-eyebrow">INDmoney · SIP discovery</div>
      <h2>Understand what makes SIP continuation difficult.</h2>
      <p>Explore the feedback in your loaded dataset, identify recurring friction signals, and open the original records before turning patterns into product hypotheses.</p>
    </div>
    """, unsafe_allow_html=True)

    n_records = len(filtered)
    n_sources = filtered["_source"].nunique()
    n_themes = filtered["_label"].nunique()
    n_dated = int(filtered["_date"].notna().sum())
    date_values = filtered["_date"].dropna()
    date_range = f"{date_values.min():%b %Y} – {date_values.max():%b %Y}" if len(date_values) else "Dates unavailable"
    kpis = [
        ("FEEDBACK RECORDS", f"{n_records:,}", "Loaded from the selected CSV"),
        ("SOURCES", f"{n_sources:,}", "Distinct source labels"),
        ("RECORDED THEMES", f"{n_themes:,}", "Distinct labels in this dataset"),
        ("DATED RECORDS", f"{n_dated:,}", date_range),
    ]
    cols = st.columns(4, gap="medium")
    for col, (label, value, note) in zip(cols, kpis):
        with col:
            st.markdown(f'<div class="dash-kpi"><div class="dash-kpi-label">{label}</div><div class="dash-kpi-value">{value}</div><div class="dash-kpi-note">{note}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="dash-section">What stands out in this dataset</div>', unsafe_allow_html=True)
    st.caption("The highlighted theme is the most frequent recorded label in the currently selected data. Frequency is a discovery signal—not proof of severity or causality.")
    label_counts = filtered["_label"].value_counts()
    if len(label_counts):
        top_label = str(label_counts.index[0])
        top_count = int(label_counts.iloc[0])
        top_share = top_count / max(1, n_records)
        feature_html = (f'<div class="dash-feature"><div class="dash-feature-title">{top_label}</div>'
                       f'<div class="dash-feature-meta"><strong>{top_count:,}</strong> records · '
                       f'<strong>{top_share:.1%}</strong> of the filtered dataset</div></div>')
        st.markdown(feature_html, unsafe_allow_html=True)

    left, right = st.columns([1.05, 1.35], gap="large")
    with left:
        st.markdown('<div class="dash-section">Where feedback comes from</div>', unsafe_allow_html=True)
        source_counts = filtered["_source"].value_counts().rename_axis("Source").reset_index(name="Records")
        source_counts["Share"] = source_counts["Records"] / max(1, n_records)
        pie = px.pie(source_counts, names="Source", values="Records", hole=0.58, title="Source mix", custom_data=["Share"])
        pie.update_traces(hovertemplate="%{label}<br>%{value:,} records (%{customdata[0]:.1%})<extra></extra>")
        st.plotly_chart(pie, use_container_width=True)
        st.dataframe(source_counts[["Source", "Records", "Share"]].style.format({"Share": "{:.1%}"}),
                     use_container_width=True, hide_index=True)
    with right:
        st.markdown('<div class="dash-section">Most common recorded themes</div>', unsafe_allow_html=True)
        theme_counts = filtered["_label"].value_counts().head(10).rename_axis("Theme").reset_index(name="Records")
        chart = px.bar(theme_counts.sort_values("Records"), x="Records", y="Theme", orientation="h",
                       title="Top 10 themes", text="Records")
        chart.update_traces(textposition="outside")
        st.plotly_chart(chart, use_container_width=True)

    st.markdown('<div class="dash-section">SIP continuity signals</div>', unsafe_allow_html=True)
    st.caption("Transparent keyword screens over original feedback. Categories can overlap; review quotes before treating these as validated findings.")
    signal_defs = {
        "Pause / stop / cancellation": ["pause", "paused", "stop sip", "stopped sip", "cancel", "cancellation", "discontinue"],
        "Payment / mandate friction": ["payment failed", "failed payment", "mandate", "autopay", "auto debit", "debit failed"],
        "Cash-flow / affordability": ["salary", "insufficient balance", "low balance", "cash flow", "emergency", "expense", "reduce sip"],
        "Returns / uncertainty": ["negative return", "loss", "market fall", "volatility", "risk", "fear", "uncertain"],
    }
    signal_rows = []
    text_series = filtered["_text"].fillna("").astype(str).str.lower()
    for signal, terms in signal_defs.items():
        mask = text_series.apply(lambda text: any(term in text for term in terms))
        count = int(mask.sum())
        signal_rows.append({"Signal": signal, "Keyword matches": count, "Share of dataset": count / max(1, n_records)})
    signal_df = pd.DataFrame(signal_rows).sort_values("Keyword matches", ascending=False)
    signal_cols = st.columns(4, gap="medium")
    for col, (_, row) in zip(signal_cols, signal_df.iterrows()):
        with col:
            st.markdown(f'<div class="dash-kpi"><div class="dash-kpi-label">{row["Signal"]}</div><div class="dash-kpi-value">{int(row["Keyword matches"]):,}</div><div class="dash-kpi-note">{row["Share of dataset"]:.1%} keyword-matched</div></div>', unsafe_allow_html=True)
    st.dataframe(signal_df, use_container_width=True, hide_index=True,
                 column_config={"Share of dataset": st.column_config.NumberColumn(format="%.1f%%")})

    if filtered["_date"].notna().any():
        st.markdown('<div class="dash-section">Feedback over time</div>', unsafe_allow_html=True)
        trend = filtered.dropna(subset=["_date"]).assign(Month=lambda x: x["_date"].dt.to_period("M").astype(str))
        trend = trend.groupby("Month").size().reset_index(name="Records")
        st.plotly_chart(px.area(trend, x="Month", y="Records", markers=True, title="Monthly feedback volume"),
                        use_container_width=True)

    st.caption("All counts are calculated from the CSV currently loaded and any active Dashboard filters. The feedback is self-selected and does not directly measure 12-month active-SIP continuation.")

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

elif page == "Research Overview":
    st.subheader("Research overview")
    st.caption("A snapshot of the evidence base and the research question.")
    n = len(data)
    date_values = data["_date"].dropna()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Feedback records", f"{n:,}")
    c2.metric("Sources", f"{data['_source'].nunique():,}")
    c3.metric("Themes / labels", f"{data['_label'].nunique():,}")
    c4.metric("Dated records", f"{len(date_values):,}")
    st.markdown("### Research objective")
    st.write("Identify and compare potential product opportunity areas that may influence the share of first-time mutual-fund investors who still have an active SIP 12 months after starting their first SIP.")
    st.info("Active SIP definition: at least one successful instalment in the 30 days around the 12-month measurement point. This feedback dataset does not directly measure that outcome.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Evidence coverage")
        src_counts = data["_source"].value_counts().rename_axis("Source").reset_index(name="Records")
        st.dataframe(src_counts, use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Most common recorded labels")
        lbl_counts = data["_label"].value_counts().head(10).rename_axis("Theme").reset_index(name="Records")
        st.dataframe(lbl_counts, use_container_width=True, hide_index=True)
    st.markdown("#### Current research questions")
    st.markdown("\\n".join(f"- {q}" for q in PRESET_QUESTIONS[:6]))
    st.caption("Counts describe this collected dataset, not all INDmoney investors.")

elif page == "Primary Research":
    st.subheader("Primary research")
    st.caption("No interview or survey records are inferred from public feedback. Add primary-research notes only when collected.")
    st.warning("Primary research data was not identified in the loaded CSV. This section intentionally does not fabricate interviews, participants, or findings.")
    with st.expander("Capture a research session note"):
        with st.form("primary_research_note"):
            participant = st.text_input("Participant code (avoid personal identifiers)")
            method = st.selectbox("Method", ["Interview", "Usability test", "Survey", "Other"])
            date = st.date_input("Research date")
            note = st.text_area("Research notes / direct observations")
            submitted = st.form_submit_button("Save note to session")
            if submitted:
                st.session_state.setdefault("primary_notes", [])
                st.session_state["primary_notes"].append({
                    "participant": participant or "Unspecified",
                    "method": method, "date": str(date), "note": note
                })
                st.success("Note saved for this app session only.")
    notes = st.session_state.get("primary_notes", [])
    if notes:
        st.markdown("### Session notes")
        for note in notes:
            st.container(border=True)
            st.write(f"{note['method']} · {note['date']} · {note['participant']}")
            st.write(note["note"])

elif page == "Contradictions & Gaps":
    st.subheader("Contradictions & gaps")
    st.caption("Automated term-based candidate detection. Treat these as review queues, not verified contradictions.")
    pairs = [
        ("Positive return / growth", ["good return", "profit", "positive return", "growth", "earning"]),
        ("Negative return / loss", ["loss", "negative return", "down", "decline", "poor return"]),
        ("Easy / smooth experience", ["easy", "smooth", "simple", "seamless"]),
        ("Difficult / broken experience", ["difficult", "problem", "issue", "failed", "error"]),
        ("Continue / long-term intent", ["continue", "long term", "keep investing", "stay invested"]),
        ("Stop / pause intent", ["stop", "cancel", "pause", "discontinue", "quit"]),
    ]
    labels = []
    for i in range(0, len(pairs), 2):
        a, b = pairs[i], pairs[i+1]
        def match_terms(terms):
            return data["_text"].str.lower().apply(lambda t: any(term in t for term in terms))
        ma, mb = match_terms(a[1]), match_terms(b[1])
        ca, cb = int(ma.sum()), int(mb.sum())
        st.markdown(f"#### {a[0]} vs {b[0]}")
        x, y = st.columns(2)
        with x:
            st.metric(a[0], ca)
            for _, row in data[ma].head(3).iterrows():
                with st.container(border=True): show_evidence(row)
        with y:
            st.metric(b[0], cb)
            for _, row in data[mb].head(3).iterrows():
                with st.container(border=True): show_evidence(row)
        st.caption("These buckets can overlap and may reflect different contexts or users; compare original records before calling this a contradiction.")
    st.markdown("### Research gaps to investigate")
    st.markdown("- Feedback is not linked to actual 12-month SIP continuation outcomes.\\n- Public feedback may overrepresent unusually positive or negative experiences.\\n- The CSV may not distinguish first-time investors from experienced investors.\\n- Reasons inferred from keyword matches require manual coding and user follow-up.")

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
    st.caption(
        "CSV-only analysis. Answers are calculated from retrieved feedback and original quotes; "
        "no external AI/API is used. Counts describe this dataset, not all INDmoney investors."
    )

    research_questions = {
        "1. Why do people start a SIP, and what do they expect to happen?": PRESET_QUESTIONS[0],
        "2. What changes in their behaviour in the months after starting?": PRESET_QUESTIONS[1],
        "3. What leads investors to pause, reduce or stop a SIP?": PRESET_QUESTIONS[2],
        "4. What do investors check, and how often, once they have started?": PRESET_QUESTIONS[3],
        "5. What confuses or worries them inside the app?": PRESET_QUESTIONS[4],
        "6. What do they do outside the app when they feel unsure?": PRESET_QUESTIONS[5],
        "7. What separates investors who continue from those who quietly drop off?": PRESET_QUESTIONS[6],
        "8. How do patterns differ across user segments?": PRESET_QUESTIONS[7],
        "9. What unmet needs emerge consistently across conversations?": PRESET_QUESTIONS[8],
        "Custom research question": "",
    }
    selected_question = st.selectbox(
        "Pre-built research questions",
        ["Choose a question…"] + list(research_questions.keys()),
        key="ask_evidence_preset"
    )
    if selected_question != "Choose a question…":
        st.session_state["ask_evidence_question"] = research_questions[selected_question]
    question = st.text_area(
        "Research question", key="ask_evidence_question", height=120,
        placeholder="Choose a preset or ask your own evidence-based research question."
    )

    if st.button("Analyze CSV evidence", type="primary", key="ask_evidence_button"):
        if not question.strip():
            st.warning("Choose a preset or enter a research question.")
        else:
            results = rank_evidence(data, question, limit=40)
            if not results:
                st.warning(
                    "No sufficiently specific records matched this question in the CSV. "
                    "Try a narrower or different wording, or inspect the Evidence Explorer. "
                    "The app will not invent an answer or pad the results with weak matches."
                )
            else:
                retrieved = [item[2] for item in results]
                matched = pd.DataFrame(retrieved)
                n_retrieved = len(matched)
                theme_counts = matched["_label"].value_counts()
                source_counts = matched["_source"].value_counts()

                st.markdown("### Answer from the CSV")
                st.write(
                    f"The question-specific search retrieved **{n_retrieved} records** "
                    f"from **{len(data):,} loaded records**. "
                    "The findings below describe only these retrieved records. "
                    "They are evidence signals, not population estimates or proof of causation."
                )

                st.markdown("### Main evidence patterns")
                profile = question_profile(question)
                pattern_defs_by_profile = {
                    "start_expectations": {
                        "Goals / future planning": ["goal", "future", "retirement", "wealth", "child", "education"],
                        "Recommendations / influence": ["recommend", "suggest", "friend", "family", "advisor", "youtube"],
                        "Starting / first investment": ["start", "started", "first", "begin", "new investor"],
                    },
                    "behaviour_change": {
                        "Monitoring / checking": ["check", "checking", "track", "tracking", "portfolio", "nav"],
                        "Contribution changes": ["increase", "decrease", "reduce", "amount", "monthly"],
                        "Payment interruptions": ["missed", "failed", "payment", "mandate", "debit"],
                        "Pause / stop behaviour": ["pause", "paused", "stop", "stopped", "cancel"],
                    },
                    "pause_reduce_stop": {
                        "Pause / restart": ["pause", "paused", "restart", "resume"],
                        "Cancellation / stopping": ["stop", "stopped", "cancel", "cancelled", "canceled", "discontinue"],
                        "Affordability / cash flow": ["salary", "balance", "expense", "emergency", "cash", "reduce", "amount"],
                        "Payment failure": ["failed", "failure", "mandate", "autopay", "debit", "missed"],
                    },
                    "monitoring": {
                        "Portfolio / return checks": ["portfolio", "return", "profit", "loss", "value"],
                        "NAV / units / status": ["nav", "unit", "status", "transaction", "pending", "allotment"],
                        "Check frequency": ["daily", "weekly", "monthly", "frequently", "every day", "check", "checking"],
                    },
                    "in_app_confusion": {
                        "Status / transaction clarity": ["status", "transaction", "pending", "nav", "unit", "allotment"],
                        "App errors / usability": ["error", "issue", "problem", "not working", "unable", "cannot"],
                        "Support / resolution": ["support", "customer care", "complaint", "resolve", "response"],
                    },
                    "outside_app": {
                        "Online research": ["youtube", "reddit", "google", "search online", "website"],
                        "Advice from others": ["friend", "friends", "family", "advisor", "adviser", "ask someone"],
                        "Community / messaging": ["whatsapp", "telegram", "forum", "community"],
                    },
                    "continuers_dropoffs": {
                        "Continuation / discipline": ["continue", "continued", "regular", "discipline", "habit", "long term"],
                        "Stop / pause / missed instalment": ["stop", "stopped", "pause", "cancel", "missed", "discontinue"],
                        "Goals / motivation": ["goal", "future", "wealth", "retirement", "education"],
                        "Financial constraints": ["salary", "income", "expense", "emergency", "balance"],
                    },
                    "segments": {
                        "Income / cash-flow references": ["salary", "income", "bonus", "business", "irregular", "payday"],
                        "Age / life-stage references": ["age", "young", "retire", "parent", "student", "child"],
                        "Influence references": ["friend", "family", "advisor", "recommend", "influenced"],
                    },
                    "unmet_needs": {
                        "Requests / missing features": ["wish", "need", "would like", "missing", "feature"],
                        "Confusion / difficulty": ["confus", "unclear", "unable", "cannot", "could not", "understand"],
                        "Support / help needs": ["help", "support", "issue", "problem", "frustrat"],
                    },
                    "affordability": {
                        "Income / balance constraints": ["salary", "payday", "income", "balance", "cash flow", "cashflow"],
                        "Unexpected expenses": ["emergency", "medical", "unexpected expense", "financial problem"],
                        "Contribution reduction / missed payment": ["reduce sip", "reduce amount", "missed", "instalment", "installment"],
                    },
                    "payments": {
                        "Payment failures": ["payment failed", "failed payment", "debit failed", "instalment failed", "installment failed"],
                        "Mandate / autopay": ["mandate", "autopay", "auto debit", "e mandate", "e-mandate"],
                        "Pending payment / transaction": ["pending", "processing", "transaction", "bank debit"],
                    },
                    "returns_risk": {
                        "Negative returns / losses": ["loss", "negative return", "market fall", "market crash", "down market"],
                        "Risk / uncertainty": ["risk", "fear", "uncertain", "volatility"],
                        "Returns / performance": ["return", "profit", "growth", "performance"],
                    },
                }
                pattern_defs = pattern_defs_by_profile.get(profile, {
                    "Question keyword matches": list(set(tokens(question)))
                })
                pattern_rows = []
                lower_text = matched["_text"].fillna("").astype(str).str.lower()
                for pattern_name, terms in pattern_defs.items():
                    mask = lower_text.apply(lambda text: any(term in text for term in terms))
                    count = int(mask.sum())
                    if count:
                        pattern_rows.append({
                            "Pattern": pattern_name,
                            "Records": count,
                            "Share of retrieved": count / n_retrieved,
                        })
                if pattern_rows:
                    pattern_df = pd.DataFrame(pattern_rows).sort_values(
                        ["Records", "Pattern"], ascending=[False, True]
                    )
                    st.dataframe(
                        pattern_df.style.format({"Share of retrieved": "{:.1%}"}),
                        use_container_width=True,
                        hide_index=True,
                    )
                    top_pattern = pattern_df.iloc[0]
                    st.markdown("**Evidence-led answer:**")
                    st.write(
                        f"Among the retrieved records, **{top_pattern['Pattern'].lower()}** "
                        f"is the most frequently detected keyword pattern "
                        f"({int(top_pattern['Records'])} of {n_retrieved} records, "
                        f"{top_pattern['Share of retrieved']:.1%}). "
                        "This is a keyword-based signal; review the original feedback below "
                        "to confirm the context and meaning."
                    )
                else:
                    st.info(
                        "The retrieved records did not match the predefined pattern screens. "
                        "Use the original feedback below for qualitative interpretation."
                    )

                st.markdown("### Retrieved themes")
                theme_summary = theme_counts.rename_axis("Theme").reset_index(name="Records")
                theme_summary["Share of retrieved"] = theme_summary["Records"] / n_retrieved
                st.dataframe(
                    theme_summary.style.format({"Share of retrieved": "{:.1%}"}),
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("### Source coverage")
                source_summary = source_counts.rename_axis("Source").reset_index(name="Records")
                st.dataframe(source_summary, use_container_width=True, hide_index=True)

                st.markdown("### Original feedback")
                st.caption(
                    "Quotes below are copied directly from the CSV. Use the source links and "
                    "record IDs to validate interpretation."
                )
                for item in results[:15]:
                    _, _, row, _, _ = item
                    with st.container(border=True):
                        show_evidence(row)

                st.markdown("### Connection to the business metric")
                st.write(
                    "These records may suggest motivations or friction relevant to SIP continuation. "
                    "The CSV does not, by itself, link feedback to first-time investor status or "
                    "12-month outcomes, so it cannot establish that these patterns cause changes "
                    "in the 12-month active-SIP continuation rate."
                )

                st.markdown("### Method and limitations")
                st.markdown(
                    f"- Retrieved evidence: **{n_retrieved}** of **{len(data):,}** loaded records.\n"
                    "- Method: question-profile keyword anchors and keyword-pattern counts; no AI-generated answer.\n"
                    "- Pattern categories can overlap; their counts should not be added together.\n"
                    "- Public feedback may be self-selected and is not necessarily representative.\n"
                    "- Treat patterns as hypotheses; manually review original quotes before using them as findings."
                )

elif page == "Challenge & Insight":
    st.subheader("Challenge & insight")
    st.caption("Use this workspace to test a proposed insight against supporting and opposing evidence.")
    insight = st.text_area("Proposed insight", placeholder="Payment uncertainty may interrupt SIP continuation.")
    terms = st.text_input("Search terms for supporting evidence", placeholder="payment, mandate, failed")
    counter_terms = st.text_input("Search terms for counter-evidence", placeholder="successful, easy, resolved")
    if st.button("Find evidence for and against", key="challenge_button"):
        def find_terms(term_string):
            ts = [t.strip().lower() for t in term_string.split(",") if t.strip()]
            mask = data["_text"].str.lower().apply(lambda text: any(t in text for t in ts)) if ts else pd.Series(False, index=data.index)
            return data[mask]
        support = find_terms(terms)
        counter = find_terms(counter_terms)
        left, right = st.columns(2)
        with left:
            st.markdown(f"### Supporting candidates ({len(support)})")
            for _, row in support.head(8).iterrows():
                with st.container(border=True): show_evidence(row)
        with right:
            st.markdown(f"### Counter-evidence candidates ({len(counter)})")
            for _, row in counter.head(8).iterrows():
                with st.container(border=True): show_evidence(row)
        st.warning("Keyword matches are candidate evidence. Manually verify relevance and whether records genuinely contradict the proposed insight.")

elif page == "Insight Methodology":
    st.subheader("Insight methodology")
    st.markdown("""
    1. **Load:** identify feedback text and available source/date/label/URL/ID fields.
    2. **Retrieve:** rank records using question-token overlap, synonym expansion, and label matches.
    3. **Count:** report retrieved records and keyword-screen counts with explicit denominators.
    4. **Interpret:** distinguish evidence from interpretation and opportunity hypotheses.
    5. **Validate:** inspect original quotes, source links, and counter-evidence; hand-label a sample.
    6. **Connect:** treat relevance to 12-month active SIP continuation as a hypothesis unless linked behavioral outcomes exist.
    """)
    st.markdown("### Retrieval scoring")
    st.code("score = 4 × direct text-token overlap + 2 × expanded overlap + 7 × direct label overlap + 4 × expanded label overlap + concept boosts")
    st.caption("This is a heuristic ranking, not a semantic model or validated classifier.")

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

elif page == "Evidence Browser":
    st.subheader("Evidence browser")
    st.caption("Use dropdowns to filter by year, platform, and assigned theme.")

    browser = data.copy()
    date_mask = browser["_date"].notna()
    if date_mask.any():
        years = sorted(browser.loc[date_mask, "_date"].dt.year.unique().tolist(), reverse=True)
    else:
        years = []

    year_options = ["All years"] + years
    platform_options = ["All platforms"] + sorted(browser["_source"].dropna().unique().tolist())
    theme_options = ["All themes"] + sorted(browser["_label"].dropna().unique().tolist())

    col_year, col_platform, col_theme = st.columns(3)
    with col_year:
        selected_year = st.selectbox("Year", year_options, key="evidence_year")
    with col_platform:
        selected_platform = st.selectbox("Platform", platform_options, key="evidence_platform")
    with col_theme:
        selected_theme = st.selectbox("Theme / label", theme_options, key="evidence_theme")

    if selected_year != "All years":
        browser = browser[browser["_date"].dt.year == int(selected_year)]
    if selected_platform != "All platforms":
        browser = browser[browser["_source"] == selected_platform]
    if selected_theme != "All themes":
        browser = browser[browser["_label"] == selected_theme]

    search = st.text_input(
        "Optional text search",
        placeholder="Search within the filtered feedback…",
        key="evidence_search",
    )
    if search.strip():
        browser = browser[
            browser["_text"].str.contains(re.escape(search.strip()), case=False, na=False)
        ]

    browser = browser.sort_values("_date", ascending=False, na_position="last")
    st.caption(f"{len(browser):,} matching records · displaying up to 100")

    if browser.empty:
        st.info("No feedback matches these dropdown selections.")
    else:
        for _, row in browser.head(100).iterrows():
            with st.container(border=True):
                show_evidence(row)
