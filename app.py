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
