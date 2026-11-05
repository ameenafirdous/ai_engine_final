        if not question.strip():
            st.warning("Choose a preset or enter a question.")
        else:
            results = rank_evidence(data, question, limit=25)
            if not results:
                st.warning(
                    "No relevant rows were found in the CSV. Try broader wording "
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
                    "retrieval, not a representative survey estimate."
                )

                if len(theme_counts):
                    st.markdown("**Most represented themes in retrieved evidence:**")
                    summary = theme_counts.head(5).rename_axis("Theme").reset_index(name="Matching records")
                    st.dataframe(summary, use_container_width=True, hide_index=True)
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
                        + ", ".join(f"{source} ({count})" for source, count in source_counts.items())
                    )

                st.markdown("#### Supporting user feedback")
                for score, idx, row in results[:12]:
                    with st.container(border=True):
                        show_evidence(row)

                st.info(
                    "Review the original quotes before reporting a conclusion. "
                    "The app summarizes retrieved CSV evidence; it does not infer "
                    "causation or invent missing information."
                )

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

