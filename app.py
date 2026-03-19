"""
Earnings Call Analyzer — Streamlit App
AI-powered first-pass analysis of earnings calls with quarter-over-quarter comparison.
"""

import streamlit as st
import plotly.graph_objects as go
import json

st.set_page_config(
    page_title="Earnings Call Analyzer",
    page_icon="📊",
    layout="wide",
)


def main():
    st.title("📊 Earnings Call Analyzer")
    st.caption(
        "AI-powered first-pass analysis with quarter-over-quarter sentiment comparison"
    )

    # Sidebar: input mode
    st.sidebar.header("Input")
    input_mode = st.sidebar.radio(
        "How would you like to load transcripts?",
        ["Ticker lookup (API)", "Upload files"],
    )

    transcripts = {}

    if input_mode == "Ticker lookup (API)":
        ticker = st.sidebar.text_input("Enter US ticker symbol", placeholder="e.g. AAPL").upper().strip()

        if st.sidebar.button("Fetch Transcripts", type="primary") and ticker:
            with st.spinner(f"Fetching transcripts for {ticker}..."):
                try:
                    from analyzer.transcript_fetcher import fetch_transcripts_by_ticker

                    results = fetch_transcripts_by_ticker(ticker)
                    if len(results) >= 2:
                        st.session_state["transcripts"] = {
                            "current_text": results[0]["content"],
                            "prior_text": results[1]["content"],
                            "current_quarter": results[0]["quarter"],
                            "prior_quarter": results[1]["quarter"],
                            "ticker": ticker,
                        }
                        st.sidebar.success(
                            f"Loaded {results[0]['quarter']} and {results[1]['quarter']}"
                        )
                    elif len(results) == 1:
                        st.sidebar.warning(
                            "Only one transcript found. Upload the prior quarter manually."
                        )
                    else:
                        st.sidebar.error(
                            f"No transcripts found for {ticker}. Try uploading instead."
                        )
                except Exception as e:
                    st.sidebar.error(f"API error: {str(e)}")

    else:
        st.sidebar.subheader("Current Quarter")
        current_file = st.sidebar.file_uploader(
            "Upload current quarter transcript",
            type=["pdf", "txt"],
            key="current",
        )
        current_label = st.sidebar.text_input(
            "Quarter label", value="Q4 2025", key="current_label"
        )

        st.sidebar.subheader("Prior Quarter")
        prior_file = st.sidebar.file_uploader(
            "Upload prior quarter transcript",
            type=["pdf", "txt"],
            key="prior",
        )
        prior_label = st.sidebar.text_input(
            "Quarter label", value="Q3 2025", key="prior_label"
        )

        if st.sidebar.button("Load Transcripts", type="primary"):
            if current_file and prior_file:
                from analyzer.transcript_fetcher import parse_uploaded_file

                with st.spinner("Parsing uploaded files..."):
                    st.session_state["transcripts"] = {
                        "current_text": parse_uploaded_file(current_file),
                        "prior_text": parse_uploaded_file(prior_file),
                        "current_quarter": current_label,
                        "prior_quarter": prior_label,
                        "ticker": "UPLOADED",
                    }
                    st.sidebar.success("Files loaded successfully.")
            else:
                st.sidebar.error("Please upload both transcripts.")

    # --- RUN ANALYSIS ---
    if "transcripts" in st.session_state:
        data = st.session_state["transcripts"]
        st.markdown(
            f"**{data.get('ticker', '')}** — "
            f"Comparing **{data['current_quarter']}** vs **{data['prior_quarter']}**"
        )

        if "report" not in st.session_state:
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                _run_analysis(data)
        else:
            if st.button("🔄 Re-run Analysis", use_container_width=True):
                del st.session_state["report"]
                _run_analysis(data)

    if "report" in st.session_state:
        _render_report(st.session_state["report"])

    elif "transcripts" not in st.session_state:
        # Landing state
        st.markdown("---")
        st.markdown(
            """
            ### How it works
            1. **Enter a ticker** or upload two earnings call transcripts
            2. **Run the analysis** — the tool extracts themes, scores sentiment, flags risks
            3. **Review the report** — interactive charts, Q&A quality tags, and follow-up questions
            4. **Export to PDF** — download a clean report

            Built with FinBERT for quantitative sentiment scoring and Claude for interpretation.
            """
        )


def _run_analysis(data: dict):
    """Run the full comparison pipeline with progress indicators."""
    from analyzer.comparison import run_comparison

    progress = st.progress(0, text="Starting analysis...")

    try:
        progress.progress(10, text="Parsing transcripts...")
        progress.progress(20, text="Extracting topics and themes...")
        progress.progress(40, text="Scoring sentiment with FinBERT...")
        progress.progress(60, text="Flagging risks...")
        progress.progress(70, text="Summarizing Q&A...")

        report = run_comparison(
            current_text=data["current_text"],
            prior_text=data["prior_text"],
            current_quarter=data["current_quarter"],
            prior_quarter=data["prior_quarter"],
        )

        progress.progress(90, text="Generating follow-up questions...")
        report["ticker"] = data.get("ticker", "")
        st.session_state["report"] = report
        progress.progress(100, text="Complete!")

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def _render_report(report: dict):
    """Render the full analysis report in Streamlit."""
    current = report["current"]
    current_q = report["current_quarter"]
    prior_q = report["prior_quarter"]

    # --- PDF EXPORT ---
    col_title, col_export = st.columns([3, 1])
    with col_export:
        try:
            from analyzer.report import generate_pdf

            pdf_bytes = generate_pdf(report)
            st.download_button(
                "📄 Export PDF",
                data=pdf_bytes,
                file_name=f"earnings_analysis_{report.get('ticker', 'report')}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.warning(f"PDF generation failed: {e}")

    st.markdown("---")

    # --- 1. KEY THEMES ---
    st.header("1. Key Themes")
    themes = current.get("themes", [])
    for t in themes:
        with st.expander(f"**{t.get('theme', 'Theme')}**", expanded=True):
            st.write(t.get("description", ""))
            st.caption(f"📌 {t.get('relevance', '')}")

    st.markdown("---")

    # --- 2. SENTIMENT HEATMAP ---
    st.header("2. Sentiment Comparison")
    shifts = report.get("sentiment_shifts", [])

    if shifts:
        topics = [s["topic"] for s in shifts]
        prior_scores = [s.get("prior_score", 0) for s in shifts]
        current_scores = [s.get("current_score", 0) for s in shifts]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=prior_q,
            x=topics,
            y=prior_scores,
            marker_color="#94a3b8",
        ))
        fig.add_trace(go.Bar(
            name=current_q,
            x=topics,
            y=current_scores,
            marker_color="#2563eb",
        ))
        fig.update_layout(
            barmode="group",
            yaxis_title="Sentiment Score",
            yaxis_range=[-1, 1],
            height=400,
            margin=dict(t=30, b=80),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="#e2e4ea")
        st.plotly_chart(fig, use_container_width=True)

        # Delta table
        delta_data = []
        for s in shifts:
            delta = s.get("delta", 0)
            delta_data.append({
                "Topic": s.get("topic", ""),
                f"{prior_q}": f"{s.get('prior_score', 0):.3f}",
                f"{current_q}": f"{s.get('current_score', 0):.3f}",
                "Delta": f"{delta:+.3f}",
            })

        st.dataframe(delta_data, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- 3. SENTIMENT NARRATIVE ---
    st.header("3. Sentiment Narrative")
    for s in shifts[:5]:
        interp = s.get("interpretation", "")
        topic = s.get("topic", "")
        delta = s.get("delta", 0)
        if interp:
            icon = "🟢" if delta > 0.05 else ("🔴" if delta < -0.05 else "⚪")
            st.markdown(f"{icon} **{topic}** ({delta:+.3f}): {interp}")

    new_topics = report.get("new_topics", [])
    dropped_topics = report.get("dropped_topics", [])
    if new_topics:
        st.info(f"**New topics in {current_q}:** {', '.join(new_topics)}")
    if dropped_topics:
        st.warning(f"**Topics dropped from {prior_q}:** {', '.join(dropped_topics)}")

    st.markdown("---")

    # --- 4. RISK FLAGS ---
    st.header("4. Risk Flags")
    risks = current.get("risks", [])
    if risks:
        for r in risks:
            severity = r.get("severity", "low").upper()
            colors_map = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "⚪"}
            icon = colors_map.get(severity, "⚪")
            st.markdown(
                f"{icon} **[{severity}] {r.get('category', '')}**: {r.get('flag', '')}"
            )
            quote = r.get("quote", "")
            if quote:
                st.caption(f'> *"{quote}"*')
    else:
        st.success("No significant risk flags identified.")

    st.markdown("---")

    # --- 5. Q&A SUMMARY ---
    st.header("5. Q&A Summary")
    qa = current.get("qa_summary", [])
    if qa:
        for q in qa:
            quality = q.get("quality", "")
            badge_colors = {
                "DIRECT ANSWER": "green",
                "PARTIAL ANSWER": "orange",
                "DEFLECTION": "red",
            }
            badge = badge_colors.get(quality, "gray")
            st.markdown(
                f"**{q.get('analyst', '')}** — {q.get('topic', '')} "
                f"&nbsp; :{badge}[{quality}]"
            )
            st.caption(q.get("summary", ""))
    else:
        st.info("Q&A section could not be parsed from the transcript.")

    st.markdown("---")

    # --- 6. QUESTIONS I'D ASK ---
    st.header("6. Questions I'd Ask")
    questions = report.get("questions", [])
    for i, q in enumerate(questions, 1):
        st.markdown(f"**{i}.** {q}")


if __name__ == "__main__":
    main()
