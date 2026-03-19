"""
PDF report generator: produces a downloadable report from the analysis.
"""

import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    HRFlowable,
)


def generate_pdf(report_data: dict) -> bytes:
    """
    Generate a PDF report from the comparison analysis data.
    Returns the PDF as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=25 * mm,
        leftMargin=25 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        spaceAfter=6,
        textColor=colors.HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=16,
        spaceAfter=8,
        textColor=colors.HexColor("#2563eb"),
    ))
    styles.add(ParagraphStyle(
        "SubHead",
        parent=styles["Heading3"],
        fontSize=11,
        spaceBefore=10,
        spaceAfter=4,
        textColor=colors.HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        "BodyText2",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "Quote",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
        leftIndent=20,
        textColor=colors.HexColor("#6a6a7a"),
        fontName="Helvetica-Oblique",
    ))

    elements = []
    current = report_data["current"]
    current_q = report_data["current_quarter"]
    prior_q = report_data["prior_quarter"]

    # --- HEADER ---
    elements.append(Paragraph("Earnings Call Analysis", styles["ReportTitle"]))
    elements.append(Paragraph(
        f"{current_q} vs {prior_q} &nbsp;|&nbsp; Generated {datetime.now().strftime('%d %b %Y')}",
        styles["BodyText2"],
    ))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e4ea")))
    elements.append(Spacer(1, 12))

    # --- KEY THEMES ---
    elements.append(Paragraph("1. Key Themes", styles["SectionHead"]))
    for t in current.get("themes", []):
        theme = t.get("theme", "")
        desc = t.get("description", "")
        elements.append(Paragraph(f"<b>{theme}</b>: {desc}", styles["BodyText2"]))
    elements.append(Spacer(1, 8))

    # --- SENTIMENT HEATMAP (as table) ---
    elements.append(Paragraph("2. Sentiment Comparison", styles["SectionHead"]))
    elements.append(Paragraph(
        "FinBERT scores per topic (-1.0 = very negative, +1.0 = very positive)",
        styles["BodyText2"],
    ))

    sentiment_table_data = [["Topic", prior_q, current_q, "Delta"]]
    shifts = report_data.get("sentiment_shifts", [])
    for s in shifts:
        topic = s.get("topic", "")
        prior_score = s.get("prior_score", 0)
        current_score = s.get("current_score", 0)
        delta = s.get("delta", 0)
        delta_str = f"{delta:+.3f}"
        sentiment_table_data.append([topic, f"{prior_score:.3f}", f"{current_score:.3f}", delta_str])

    if len(sentiment_table_data) > 1:
        t = Table(sentiment_table_data, colWidths=[150, 80, 80, 80])
        style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e4ea")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ])
        # Color-code delta column
        for i in range(1, len(sentiment_table_data)):
            delta_val = sentiment_table_data[i][3]
            try:
                d = float(delta_val)
                if d > 0.05:
                    style.add("TEXTCOLOR", (3, i), (3, i), colors.HexColor("#16a34a"))
                elif d < -0.05:
                    style.add("TEXTCOLOR", (3, i), (3, i), colors.HexColor("#dc2626"))
            except ValueError:
                pass
        t.setStyle(style)
        elements.append(t)
    elements.append(Spacer(1, 8))

    # --- SENTIMENT NARRATIVE ---
    elements.append(Paragraph("3. Sentiment Narrative", styles["SectionHead"]))
    for s in shifts[:5]:
        interp = s.get("interpretation", "")
        topic = s.get("topic", "")
        if interp:
            elements.append(Paragraph(f"<b>{topic}</b>: {interp}", styles["BodyText2"]))
    elements.append(Spacer(1, 8))

    # New / dropped topics
    new_topics = report_data.get("new_topics", [])
    dropped_topics = report_data.get("dropped_topics", [])
    if new_topics:
        elements.append(Paragraph(
            f"<b>New topics in {current_q}:</b> {', '.join(new_topics)}",
            styles["BodyText2"],
        ))
    if dropped_topics:
        elements.append(Paragraph(
            f"<b>Topics dropped from {prior_q}:</b> {', '.join(dropped_topics)}",
            styles["BodyText2"],
        ))
    elements.append(Spacer(1, 8))

    # --- RISK FLAGS ---
    elements.append(Paragraph("4. Risk Flags", styles["SectionHead"]))
    risks = current.get("risks", [])
    if risks:
        for r in risks:
            severity = r.get("severity", "").upper()
            category = r.get("category", "")
            flag = r.get("flag", "")
            quote = r.get("quote", "")
            sev_color = {"HIGH": "#dc2626", "MEDIUM": "#d97706", "LOW": "#6a6a7a"}.get(severity, "#6a6a7a")
            elements.append(Paragraph(
                f'<font color="{sev_color}"><b>[{severity}]</b></font> '
                f"<b>{category}</b>: {flag}",
                styles["BodyText2"],
            ))
            if quote:
                elements.append(Paragraph(f'"{quote}"', styles["Quote"]))
    else:
        elements.append(Paragraph("No significant risk flags identified.", styles["BodyText2"]))
    elements.append(Spacer(1, 8))

    # --- Q&A SUMMARY ---
    elements.append(Paragraph("5. Q&A Summary", styles["SectionHead"]))
    qa = current.get("qa_summary", [])
    if qa:
        qa_table_data = [["Analyst", "Topic", "Summary", "Quality"]]
        for q in qa:
            quality = q.get("quality", "")
            q_color = {
                "DIRECT ANSWER": "#16a34a",
                "PARTIAL ANSWER": "#d97706",
                "DEFLECTION": "#dc2626",
            }.get(quality, "#6a6a7a")
            qa_table_data.append([
                Paragraph(q.get("analyst", "")[:30], styles["BodyText2"]),
                Paragraph(q.get("topic", ""), styles["BodyText2"]),
                Paragraph(q.get("summary", ""), styles["BodyText2"]),
                Paragraph(f'<font color="{q_color}"><b>{quality}</b></font>', styles["BodyText2"]),
            ])
        qt = Table(qa_table_data, colWidths=[90, 70, 200, 80])
        qt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e4ea")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ]))
        elements.append(qt)
    else:
        elements.append(Paragraph("Q&A section could not be parsed.", styles["BodyText2"]))
    elements.append(Spacer(1, 8))

    # --- QUESTIONS I'D ASK ---
    elements.append(Paragraph("6. Questions I'd Ask", styles["SectionHead"]))
    questions = report_data.get("questions", [])
    for i, q in enumerate(questions, 1):
        elements.append(Paragraph(f"<b>{i}.</b> {q}", styles["BodyText2"]))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.read()
