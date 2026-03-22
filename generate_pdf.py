#!/usr/bin/env python3
"""Generate PMIS Memory System Documentation PDF."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import os

OUTPUT = os.path.expanduser("~/Desktop/memory/PMIS_Memory_System_Documentation.pdf")

# ── Brand Colors ──
DARK_BG = HexColor("#0c0c0f")
CARD_BG = HexColor("#16161b")
GREEN = HexColor("#4ade80")
BLUE = HexColor("#60a5fa")
PURPLE = HexColor("#a78bfa")
ORANGE = HexColor("#fb923c")
RED = HexColor("#f87171")
TEAL = HexColor("#2dd4bf")
YELLOW = HexColor("#facc15")
FG = HexColor("#e2e0d8")
FG2 = HexColor("#8a8880")
BORDER = HexColor("#2a2a30")
# Light theme for print
PRIMARY = HexColor("#0a0a0a")
ACCENT = HexColor("#2563EB")
ACCENT2 = HexColor("#16a34a")
LIGHT_BG = HexColor("#f5f6f8")
LIGHT_BORDER = HexColor("#d0d0d0")
GRAY = HexColor("#666666")

W, H = A4

# ── Page Template ──
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def _draw_page_decorations(self, num_pages):
        page_num = self._pageNumber
        # Footer line
        self.setStrokeColor(LIGHT_BORDER)
        self.setLineWidth(0.5)
        self.line(50, 40, W - 50, 40)
        # Page number
        self.setFont("Helvetica", 8)
        self.setFillColor(GRAY)
        self.drawCentredString(W / 2, 28, f"Page {page_num} of {num_pages}")
        # Footer text
        self.drawString(50, 28, "PMIS Documentation")
        self.drawRightString(W - 50, 28, "Yantra AI Labs")


# ── Styles ──
styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    'DocTitle', parent=styles['Title'],
    fontSize=26, leading=32, textColor=PRIMARY,
    fontName='Helvetica-Bold', spaceAfter=6, alignment=TA_CENTER
))
styles.add(ParagraphStyle(
    'DocSubtitle', parent=styles['Normal'],
    fontSize=13, leading=18, textColor=GRAY,
    fontName='Helvetica', spaceAfter=30, alignment=TA_CENTER
))
styles.add(ParagraphStyle(
    'H1', parent=styles['Heading1'],
    fontSize=20, leading=26, textColor=PRIMARY,
    fontName='Helvetica-Bold', spaceBefore=28, spaceAfter=12,
    borderWidth=0, borderPadding=0
))
styles.add(ParagraphStyle(
    'H2', parent=styles['Heading2'],
    fontSize=15, leading=20, textColor=ACCENT,
    fontName='Helvetica-Bold', spaceBefore=20, spaceAfter=8
))
styles.add(ParagraphStyle(
    'H3', parent=styles['Heading3'],
    fontSize=12, leading=16, textColor=HexColor("#333333"),
    fontName='Helvetica-Bold', spaceBefore=14, spaceAfter=6
))
styles.add(ParagraphStyle(
    'Body', parent=styles['Normal'],
    fontSize=10, leading=15, textColor=HexColor("#222222"),
    fontName='Helvetica', spaceAfter=8, alignment=TA_JUSTIFY
))
styles.add(ParagraphStyle(
    'BodyBold', parent=styles['Normal'],
    fontSize=10, leading=15, textColor=PRIMARY,
    fontName='Helvetica-Bold', spaceAfter=4
))
styles.add(ParagraphStyle(
    'CodeBlock', parent=styles['Normal'],
    fontSize=8.5, leading=12, textColor=HexColor("#1a1a2e"),
    fontName='Courier', spaceAfter=4, spaceBefore=2,
    leftIndent=16, backColor=HexColor("#f0f0f5"),
    borderWidth=0.5, borderColor=LIGHT_BORDER, borderPadding=6,
    borderRadius=4
))
styles.add(ParagraphStyle(
    'TOCEntry', parent=styles['Normal'],
    fontSize=11, leading=20, textColor=PRIMARY,
    fontName='Helvetica', leftIndent=10
))
styles.add(ParagraphStyle(
    'TOCSection', parent=styles['Normal'],
    fontSize=12, leading=22, textColor=ACCENT,
    fontName='Helvetica-Bold', leftIndent=0, spaceBefore=6
))
styles.add(ParagraphStyle(
    'Caption', parent=styles['Normal'],
    fontSize=9, leading=12, textColor=GRAY,
    fontName='Helvetica-Oblique', spaceAfter=10, alignment=TA_CENTER
))
styles.add(ParagraphStyle(
    'BulletItem', parent=styles['Normal'],
    fontSize=10, leading=15, textColor=HexColor("#222222"),
    fontName='Helvetica', spaceAfter=4, leftIndent=20, bulletIndent=10
))

def bullet(text):
    return Paragraph(f"<bullet>&bull;</bullet> {text}", styles['BulletItem'])

def code_block(lines):
    text = "<br/>".join(lines)
    return Paragraph(text, styles['CodeBlock'])

def hr():
    return HRFlowable(width="100%", thickness=0.5, color=LIGHT_BORDER, spaceBefore=10, spaceAfter=10)

def make_table(headers, rows, col_widths=None):
    data = [headers] + rows
    if col_widths is None:
        col_widths = [W * 0.85 / len(headers)] * len(headers)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('LEADING', (0, 0), (-1, -1), 14),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT_BG]),
        ('GRID', (0, 0), (-1, -1), 0.5, LIGHT_BORDER),
    ]))
    return t

def make_dark_table(headers, rows, col_widths=None):
    """Accent table with darker header."""
    data = [headers] + rows
    if col_widths is None:
        col_widths = [W * 0.85 / len(headers)] * len(headers)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('LEADING', (0, 0), (-1, -1), 14),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT_BG]),
        ('GRID', (0, 0), (-1, -1), 0.5, LIGHT_BORDER),
    ]))
    return t


# ══════════════════════════════════════════════
# BUILD DOCUMENT
# ══════════════════════════════════════════════

story = []

# ── COVER PAGE ──
story.append(Spacer(1, 120))
story.append(Paragraph("PMIS", ParagraphStyle(
    'CoverPMIS', parent=styles['Title'],
    fontSize=60, leading=70, textColor=ACCENT,
    fontName='Helvetica-Bold', alignment=TA_CENTER
)))
story.append(Spacer(1, 8))
story.append(Paragraph("Personal Memory Intelligence System", ParagraphStyle(
    'CoverSub1', parent=styles['Normal'],
    fontSize=18, leading=24, textColor=PRIMARY,
    fontName='Helvetica', alignment=TA_CENTER
)))
story.append(Spacer(1, 30))
story.append(HRFlowable(width="40%", thickness=2, color=ACCENT, spaceBefore=0, spaceAfter=0))
story.append(Spacer(1, 30))
story.append(Paragraph("Complete Technical Documentation", ParagraphStyle(
    'CoverSub2', parent=styles['Normal'],
    fontSize=14, leading=20, textColor=GRAY,
    fontName='Helvetica', alignment=TA_CENTER
)))
story.append(Paragraph("Memory Structuring Rules &amp; Component Reference", ParagraphStyle(
    'CoverSub3', parent=styles['Normal'],
    fontSize=12, leading=18, textColor=GRAY,
    fontName='Helvetica', alignment=TA_CENTER, spaceAfter=60
)))
story.append(Spacer(1, 100))
story.append(Paragraph("Yantra AI Labs", ParagraphStyle(
    'CoverBrand', parent=styles['Normal'],
    fontSize=11, leading=16, textColor=GRAY,
    fontName='Helvetica-Bold', alignment=TA_CENTER, letterSpacing=3
)))
story.append(Paragraph("March 2026 &bull; v3 Architecture", ParagraphStyle(
    'CoverDate', parent=styles['Normal'],
    fontSize=10, leading=14, textColor=LIGHT_BORDER,
    fontName='Helvetica', alignment=TA_CENTER
)))
story.append(PageBreak())

# ── TABLE OF CONTENTS ──
story.append(Paragraph("Table of Contents", styles['H1']))
story.append(hr())

toc_items = [
    ("Part I", "Memory Structuring Rules - Plain English", True),
    ("1.1", "The 3-Level Hierarchy", False),
    ("1.2", "The 4 Structuring Decisions", False),
    ("1.3", "The Memory Lifecycle", False),
    ("Part II", "Technical Components", True),
    ("2.1", "Database Schema", False),
    ("2.2", "The Dedup Engine", False),
    ("2.3", "The Temporal Engine", False),
    ("2.4", "Evidence-Based Weight Engine", False),
    ("2.5", "Anchor Discrimination Power (ADP)", False),
    ("2.6", "Mode Vectors", False),
    ("2.7", "Transfer Edges", False),
    ("2.8", "Winning Structure Snapshots", False),
    ("2.9", "Decision Convergence", False),
    ("2.10", "Full Retrieval Ranking Formula", False),
    ("Part III", "Current System State", True),
]
for num, title, is_section in toc_items:
    style = styles['TOCSection'] if is_section else styles['TOCEntry']
    prefix = f"<b>{num}</b>" if is_section else f"&nbsp;&nbsp;{num}"
    story.append(Paragraph(f"{prefix}&nbsp;&nbsp;&nbsp;{title}", style))

story.append(PageBreak())

# ══════════════════════════════════════════════
# PART I: PLAIN ENGLISH
# ══════════════════════════════════════════════

story.append(Paragraph("Part I: Memory Structuring Rules", styles['H1']))
story.append(Paragraph("How the system works, explained simply.", styles['Caption']))
story.append(hr())

# 1.1 Hierarchy
story.append(Paragraph("1.1 &nbsp; The 3-Level Hierarchy", styles['H2']))
story.append(Paragraph(
    "Think of the memory system like a <b>library you build as you learn</b>. "
    "Knowledge is organized in three levels, from broadest to most specific:",
    styles['Body']
))

story.append(make_table(
    ["Level", "Analogy", "What It Represents"],
    [
        ["Super Context", '"Which shelf in the library?"', "A broad, reusable work domain"],
        ["Context", '"Which chapter on that shelf?"', "A distinct skill or phase within the domain"],
        ["Anchor", '"Which specific fact in that chapter?"', "One atomic, reusable insight"],
    ],
    col_widths=[90, 160, 230]
))
story.append(Spacer(1, 10))

story.append(Paragraph("<b>Real example from the system:</b>", styles['Body']))
story.append(code_block([
    "Security Demo Page                         &lt;-- Super Context (the shelf)",
    "    Track Record section                    &lt;-- Context (the chapter)",
    "        Cards use flex:1 for equal sizing   &lt;-- Anchor (the fact)",
    "        IntersectionObserver for scroll reveal",
    "    Footer navigation",
    "        GPU Calculator link in footer",
]))
story.append(Paragraph(
    "<b>The rule:</b> Super Contexts are reusable domains, Contexts are skills/phases within them, "
    "Anchors are single atomic facts with numbers and reasoning.",
    styles['Body']
))

# 1.2 Structuring Decisions
story.append(Paragraph("1.2 &nbsp; The 4 Structuring Decisions", styles['H2']))
story.append(Paragraph(
    "Every time knowledge gets stored, four structuring choices are made. "
    "These decisions determine whether memory is useful or noise:",
    styles['Body']
))

story.append(make_table(
    ["#", "Decision", "Rule", "Bad vs. Good Example"],
    [
        ["1", "Super Context\nname", "Broad enough to\nreuse across tasks",
         'BAD: "Tuesday\'s email"\nGOOD: "B2B Cold Outreach"'],
        ["2", "Context\ngrouping", "Group by skill,\nnot by date",
         'BAD: "Mar 18 session"\nGOOD: "Email copywriting"'],
        ["3", "Anchor\nspecificity", "One fact, with\nnumbers and WHY",
         'BAD: "Good email copy"\nGOOD: "Short subjects get 40% higher open rates"'],
        ["4", "Weight\nassignment", "0.9 = essential\n0.5 = moderate\n0.3 = minor",
         'Ask: "If I could only remember 2 things, which would they be?" Those get 0.8-0.9'],
    ],
    col_widths=[25, 70, 110, 275]
))

# 1.3 Lifecycle
story.append(Paragraph("1.3 &nbsp; The Memory Lifecycle", styles['H2']))
story.append(Paragraph(
    "Every conversation follows the same four-step cycle. The system automatically retrieves "
    "before work and stores after work. Scoring is optional but dramatically improves quality over time.",
    styles['Body']
))

story.append(code_block([
    "Conversation starts",
    "       |",
    "       v",
    "  RETRIEVE ---&gt; 'What do I already know about this?'",
    "       |",
    "       v",
    "    DO WORK ---&gt; Use retrieved context to do better work",
    "       |",
    "       v",
    "    STORE -----&gt; Extract key learnings, structure them, save",
    "       |",
    "       v",
    "   (SCORE) ----&gt; 'Did it work?' Score adjusts weights over time",
]))

story.append(Paragraph(
    "<b>Auto-retrieve</b> happens before every task. The system searches all Super Contexts using "
    "word overlap, quality scores, temporal boosts, and mode vector similarity. "
    "Top 3 matches are returned with their full anchor trees.",
    styles['Body']
))
story.append(Paragraph(
    "<b>Auto-store</b> happens after completing work. Claude structures the learnings into the "
    "hierarchy (making the 4 decisions above) and saves them. The dedup engine ensures "
    "repeated stores merge into one growing tree rather than creating duplicates.",
    styles['Body']
))
story.append(Paragraph(
    "<b>Scoring</b> is the feedback loop. When a task is scored (1-5), the system captures a "
    "structure snapshot, recomputes evidence-based weights, and updates quality metrics. "
    "Without scoring, weights remain at initial estimates forever.",
    styles['Body']
))

story.append(PageBreak())

# ══════════════════════════════════════════════
# PART II: TECHNICAL COMPONENTS
# ══════════════════════════════════════════════

story.append(Paragraph("Part II: Technical Components", styles['H1']))
story.append(Paragraph("How each engine works under the hood.", styles['Caption']))
story.append(hr())

# 2.1 Schema
story.append(Paragraph("2.1 &nbsp; Database Schema", styles['H2']))
story.append(Paragraph(
    "SQLite database with <b>zero external dependencies</b> (Python stdlib only). "
    "Five tables store all knowledge, relationships, tasks, and decisions.",
    styles['Body']
))

story.append(Paragraph("<b>nodes</b> - Every piece of knowledge is a node", styles['BodyBold']))
story.append(make_table(
    ["Column", "Type", "Purpose"],
    [
        ["id", "TEXT", "Random 10-char hex (uuid4)"],
        ["type", "TEXT", "super_context | context | anchor"],
        ["title", "TEXT", "Human-readable name; also the dedup key"],
        ["content", "TEXT", "Actual knowledge (meaningful for anchors only)"],
        ["weight", "REAL", "Current weight (0-1). Starts as guess, evolves with evidence"],
        ["initial_weight", "REAL", "Original guess, preserved forever for blending"],
        ["use_count", "INT", "How many times this node has been touched"],
        ["quality", "REAL", "For SCs only: running avg of task scores (0-5)"],
        ["occurrence_log", "TEXT", 'JSON array of dates: ["2026-03-15", ...]'],
        ["recency", "REAL", "Exponential decay from last use (half-life 7 days)"],
        ["frequency", "REAL", "Uses in last 30 days / 30"],
        ["consistency", "REAL", "1 / (std_dev of gaps + 1)"],
        ["memory_stage", "TEXT", "impulse | active | established | fading"],
        ["discrimination_power", "REAL", "ADP: var(anchor scores) / var(all scores)"],
        ["mode_vector", "TEXT", 'JSON: {"work": 0.8, "creative": 0.3, ...}'],
    ],
    col_widths=[100, 45, 335]
))
story.append(Spacer(1, 8))

story.append(Paragraph("<b>edges</b> - Links between nodes", styles['BodyBold']))
story.append(make_table(
    ["Column", "Type", "Purpose"],
    [
        ["src, tgt", "TEXT", "Parent to Child (SC to CTX or CTX to ANC)"],
        ["type", "TEXT", "parent_child (hierarchy) or transfer (cross-domain)"],
        ["weight", "REAL", "Edge strength (usually mirrors child weight)"],
    ],
    col_widths=[100, 45, 335]
))
story.append(Spacer(1, 8))

story.append(Paragraph("<b>tasks</b> - Every store operation creates a task log", styles['BodyBold']))
story.append(make_table(
    ["Column", "Type", "Purpose"],
    [
        ["id", "TEXT", "The task_id needed for scoring later"],
        ["sc_id", "TEXT", "Which Super Context this belongs to"],
        ["score", "REAL", "0 until scored, then 1.0-5.0"],
        ["structure_snapshot", "TEXT", "JSON snapshot of exact tree at score time"],
    ],
    col_widths=[120, 45, 315]
))
story.append(Spacer(1, 8))

story.append(Paragraph("<b>task_anchors</b> - Junction: which anchors were present in which task", styles['BodyBold']))
story.append(Paragraph(
    "This is what makes evidence-based weights work. When you score task X as 4.5, "
    "the system looks up every anchor linked to task X and credits them.",
    styles['Body']
))

story.append(Paragraph("<b>decisions</b> - Explicit decisions that should not be contradicted", styles['BodyBold']))
story.append(make_table(
    ["Column", "Type", "Purpose"],
    [
        ["decision", "TEXT", "The actual decision text"],
        ["confidence", "REAL", "0-1, grows with high scores, decays with low"],
        ["reversible", "BOOL", "Locked decisions cannot be overridden"],
    ],
    col_widths=[100, 45, 335]
))

story.append(PageBreak())

# 2.2 Dedup
story.append(Paragraph("2.2 &nbsp; The Dedup Engine", styles['H2']))
story.append(Paragraph(
    "The single most important mechanism. Without it, you would get 50 copies of "
    "'Security Demo Page'. Dedup ensures all stores to the same domain merge into one growing tree.",
    styles['Body']
))

story.append(Paragraph("<b>Two-phase matching:</b>", styles['BodyBold']))
story.append(bullet("<b>Phase 1 - Exact match:</b> If title strings are identical, reuse the node immediately."))
story.append(bullet("<b>Phase 2 - Fuzzy Jaccard:</b> Extract significant words (strip stop words, require length &gt; 2). "
    "Two words 'match' if: identical, one contains the other, or they share a 4+ character prefix. "
    "Compute Jaccard = matched_pairs / union_size. If Jaccard &ge; 0.35, treat as same node."))
story.append(Spacer(1, 6))

story.append(Paragraph("<b>When reusing a node:</b>", styles['BodyBold']))
story.append(bullet("Bump <b>use_count</b>"))
story.append(bullet("Record today in <b>occurrence_log</b> (feeds temporal engine)"))
story.append(bullet("Keep whichever <b>weight</b> is higher (old vs new)"))

story.append(code_block([
    "Example:",
    '  "Email copywriting tips"  --&gt;  ["email", "copywriting", "tips"]',
    '  "Copywriting for emails"  --&gt;  ["copywriting", "emails"]',
    "",
    '  "email" fuzzy-matches "emails" (containment)',
    '  "copywriting" exact match',
    "  Jaccard = 2 matched / 3 union = 0.67 &gt;= 0.35  --&gt;  SAME NODE",
]))

# 2.3 Temporal
story.append(Paragraph("2.3 &nbsp; The Temporal Engine", styles['H2']))
story.append(Paragraph(
    "Classifies every node into one of 4 memory stages based on real usage patterns. "
    "This determines how prominently a memory surfaces during retrieval.",
    styles['Body']
))

story.append(Paragraph("<b>Three raw signals (computed from occurrence_log):</b>", styles['BodyBold']))
story.append(make_table(
    ["Signal", "Formula", "Intuition"],
    [
        ["Recency", "e^(-0.693 x days / 7)", "Half-life of 7 days. Yesterday = 0.91, week ago = 0.50, month ago = 0.05"],
        ["Frequency", "uses_in_30_days / 30", "15 uses in 30 days = 0.50, 3 uses = 0.10"],
        ["Consistency", "1 / (std_dev_of_gaps + 1)", "Every 3 days (std=0) = 1.0, erratic (std=10) = 0.09"],
    ],
    col_widths=[80, 140, 260]
))
story.append(Spacer(1, 8))

story.append(Paragraph("<b>Stage classification &amp; retrieval boost:</b>", styles['BodyBold']))
story.append(make_dark_table(
    ["Stage", "Condition", "Boost", "Meaning"],
    [
        ["ESTABLISHED", "freq >= 0.15 AND consistency > 0.3", "x 1.5", "Battle-tested, proven over time"],
        ["ACTIVE", "recency > 0.3 AND freq >= 0.05", "x 1.2", "Currently in use, hot knowledge"],
        ["IMPULSE", "Default / new", "x 0.8", "New, unproven, slight penalty"],
        ["FADING", "recency < 0.3 AND freq < 0.05", "x 0.5", "Old and forgotten, deprioritized"],
    ],
    col_widths=[85, 180, 50, 165]
))

# 2.4 Evidence Weights
story.append(Paragraph("2.4 &nbsp; Evidence-Based Weight Engine", styles['H2']))
story.append(Paragraph(
    "Weights start as Claude's initial estimate. Over time, they evolve based on actual task outcomes. "
    "The more data, the more evidence dominates over the initial guess.",
    styles['Body']
))

story.append(Paragraph("<b>The blending formula:</b>", styles['BodyBold']))
story.append(code_block([
    "raw_evidence = (mean_task_score / 5.0) ^ 0.7   # Non-linear power curve",
    "",
    "blended = initial * decay + evidence * (1 - decay)",
    "",
    "Decay schedule (how much to trust initial guess):",
    "  n=1 scored task:   decay = 0.70  (70% initial, 30% evidence)",
    "  n=2:               decay = 0.50  (50/50)",
    "  n=3-4:             decay = 0.30  (30% initial, 70% evidence)",
    "  n=5+:              decay = 0.15  (15% initial, 85% evidence)",
]))

story.append(Paragraph("<b>Worked example:</b>", styles['BodyBold']))
story.append(Paragraph(
    "Initial guess = 0.6. Anchor appears in 3 tasks scored 4.5, 4.0, 3.5. "
    "Mean = 4.0. Evidence = (4.0/5.0)<super>0.7</super> = 0.86. "
    "At n=3: blended = 0.3 x 0.6 + 0.7 x 0.86 = <b>0.78</b>. "
    "The weight rose because evidence proved it works.",
    styles['Body']
))

# 2.5 ADP
story.append(Paragraph("2.5 &nbsp; Anchor Discrimination Power (ADP)", styles['H2']))
story.append(Paragraph(
    "Answers: <b>'Does HOW you execute this anchor matter, or does it reliably work regardless?'</b>",
    styles['Body']
))
story.append(code_block([
    "ADP = variance(scores of tasks with this anchor) / variance(all task scores)",
]))
story.append(make_table(
    ["ADP Value", "Meaning", "Action"],
    [
        ["< 0.5", "Reliable regardless", "This anchor just works. Use it confidently."],
        ["0.5 - 1.0", "Normal variance", "Standard behavior, no special handling."],
        ["> 1.0", "Execution-sensitive", "HOW matters. Show best/worst execution params."],
    ],
    col_widths=[80, 140, 260]
))
story.append(Paragraph(
    "When ADP &gt; 1.0, retrieval includes the best and worst execution parameters "
    "from task_anchors so you know what approach worked vs. what failed.",
    styles['Body']
))

story.append(PageBreak())

# 2.6 Mode Vectors
story.append(Paragraph("2.6 &nbsp; Mode Vectors", styles['H2']))
story.append(Paragraph(
    "Every Super Context gets a 6-dimensional 'personality' vector, inferred from title keywords. "
    "During retrieval, cosine similarity between the query's mode and each SC's mode provides "
    "an additional ranking signal.",
    styles['Body']
))

story.append(Paragraph("<b>The 6 dimensions:</b>", styles['BodyBold']))
story.append(code_block([
    '{"work": 0.8, "creative": 0.3, "social": 0.5, "learning": 0.1, "travel": 0.0, "home": 0.0}',
]))

story.append(Paragraph("<b>Template examples:</b>", styles['BodyBold']))
story.append(make_table(
    ["Keyword", "work", "creative", "social", "learning", "travel", "home"],
    [
        ["outreach", "0.9", "0.2", "0.5", "0.1", "0.0", "0.0"],
        ["website", "0.7", "0.8", "0.1", "0.2", "0.0", "0.1"],
        ["security", "0.8", "0.3", "0.1", "0.5", "0.0", "0.1"],
        ["research", "0.4", "0.2", "0.1", "0.9", "0.0", "0.2"],
        ["design", "0.6", "0.9", "0.1", "0.3", "0.0", "0.2"],
    ],
    col_widths=[80, 60, 60, 60, 60, 60, 60]
))
story.append(Paragraph(
    "A security question retrieves security-domain memories before marketing memories, "
    "even if word overlap is similar, because the mode vectors are closer.",
    styles['Body']
))

# 2.7 Transfer Edges
story.append(Paragraph("2.7 &nbsp; Transfer Edges", styles['H2']))
story.append(Paragraph(
    "Cross-pollination between Super Contexts. If two SCs share similar internal structure "
    "(contexts and anchors), they are linked with bidirectional transfer edges.",
    styles['Body']
))
story.append(Paragraph("<b>Detection:</b>", styles['BodyBold']))
story.append(bullet("Compare context titles between two SCs using Jaccard word overlap"))
story.append(bullet("Compare anchor titles across both trees"))
story.append(bullet("Structural similarity = 0.6 x context_sim + 0.4 x anchor_overlap"))
story.append(bullet("If similarity &ge; 0.25, create bidirectional transfer edge"))
story.append(Spacer(1, 4))
story.append(Paragraph("<b>During retrieval:</b>", styles['BodyBold']))
story.append(Paragraph(
    "If SC-A matches your query, its transfer partners are also returned at <b>60% discount</b> "
    "on relevance. They appear in results but don't dominate over the primary match.",
    styles['Body']
))

# 2.8 Winning Snapshots
story.append(Paragraph("2.8 &nbsp; Winning Structure Snapshots", styles['H2']))
story.append(Paragraph(
    "When you score a task, the system captures the <b>exact tree</b> that produced that outcome:",
    styles['Body']
))
story.append(code_block([
    "{",
    '  "Email copywriting": ["Short subjects", "Threat-intel language"],',
    '  "Target research": ["VP Security &gt; CISO"]',
    "}",
]))
story.append(Paragraph(
    "On future retrieval, anchors marked <b>in_winning_structure: true</b> are surfaced first. "
    "They are the proven recipe. The highest-scoring task per SC defines the winning structure.",
    styles['Body']
))

# 2.9 Decision Convergence
story.append(Paragraph("2.9 &nbsp; Decision Convergence", styles['H2']))
story.append(Paragraph(
    "For long-running projects, decisions accumulate and the system tracks how 'decided' "
    "a domain is. This prevents AI context drift across conversations.",
    styles['Body']
))

story.append(Paragraph("<b>Confidence dynamics:</b>", styles['BodyBold']))
story.append(make_table(
    ["Event", "Effect", "Formula"],
    [
        ["High task score (>= 4.0)", "Confidence grows", "conf += 0.1 x (1.0 - conf)"],
        ["Low task score (< 2.5)", "Confidence decays, decision unlocks", "conf *= 0.7, reversible = true"],
    ],
    col_widths=[140, 140, 200]
))
story.append(Spacer(1, 6))
story.append(Paragraph("<b>Convergence score:</b>", styles['BodyBold']))
story.append(code_block([
    "convergence = 0.4 x avg_confidence + 0.4 x lock_ratio + 0.2 x min(total_decisions/10, 1.0)",
]))
story.append(Paragraph(
    "Higher convergence means the system is more 'decided' about this domain. "
    "Locked decisions are presented as constraints that should not be contradicted. "
    "Open decisions can be revisited with new evidence.",
    styles['Body']
))

story.append(PageBreak())

# 2.10 Retrieval Formula
story.append(Paragraph("2.10 &nbsp; Full Retrieval Ranking Formula", styles['H2']))
story.append(Paragraph(
    "When you search, each Super Context gets a composite relevance score:",
    styles['Body']
))
story.append(code_block([
    "score = (word_overlap + quality_bonus + use_bonus) x temporal_boost x mode_boost",
]))

story.append(make_table(
    ["Factor", "Calculation", "Max Contribution"],
    [
        ["word_overlap", "Matching words across SC title, contexts, anchors\n(anchors count 0.5 each)", "Unbounded"],
        ["quality_bonus", "SC quality / 5 x 2", "+2.0"],
        ["use_bonus", "min(use_count / 10, 0.5)", "+0.5"],
        ["temporal_boost", "0.5x (fading) to 1.5x (established)", "Multiplier"],
        ["mode_boost", "0.5 + cosine_similarity(query_mode, sc_mode)", "Multiplier"],
    ],
    col_widths=[100, 240, 140]
))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Top 3 SCs are returned with full context trees. Anchors within each context are sorted by: "
    "(1) winning-structure membership first, then (2) evidence-based weight descending.",
    styles['Body']
))

story.append(PageBreak())

# ══════════════════════════════════════════════
# PART III: CURRENT STATE
# ══════════════════════════════════════════════

story.append(Paragraph("Part III: Current System State", styles['H1']))
story.append(Paragraph("As of March 2026.", styles['Caption']))
story.append(hr())

story.append(make_dark_table(
    ["Metric", "Value", "Notes"],
    [
        ["Super Contexts", "12", "Broad work domains stored"],
        ["Contexts", "71", "Skills/phases across all domains"],
        ["Anchors", "231", "Atomic reusable insights"],
        ["Edges", "302", "Parent-child + transfer links"],
        ["Tasks Logged", "23", "Store operations completed"],
        ["Tasks Scored", "0", "BIGGEST GAP - no feedback loop active"],
        ["Avg Score", "0", "No scores yet"],
        ["Memory Stages", "All impulse", "No node has graduated - need repeated usage"],
    ],
    col_widths=[120, 80, 280]
))

story.append(Spacer(1, 16))
story.append(Paragraph("<b>Key Observation:</b>", styles['BodyBold']))
story.append(Paragraph(
    "The system is accumulating knowledge (231 anchors across 12 domains) but has not entered "
    "the feedback loop. Zero tasks have been scored, meaning all weights are still initial guesses. "
    "All 314 nodes remain in the 'impulse' stage because none have been reused across multiple sessions.",
    styles['Body']
))
story.append(Spacer(1, 8))
story.append(Paragraph("<b>To activate the full system:</b>", styles['BodyBold']))
story.append(bullet("Score completed tasks (even retroactively) to start evidence-based weight evolution"))
story.append(bullet("Reuse Super Contexts across conversations so nodes graduate from impulse to active"))
story.append(bullet("Run <font face='Courier' size='9'>python3 scripts/memory.py rebuild</font> periodically to refresh temporal scores and detect transfers"))

story.append(Spacer(1, 30))
story.append(hr())
story.append(Paragraph(
    "Document generated by PMIS v3 &bull; Yantra AI Labs &bull; March 2026",
    ParagraphStyle('Footer', parent=styles['Normal'],
        fontSize=9, textColor=GRAY, fontName='Helvetica', alignment=TA_CENTER)
))


# ── BUILD ──
doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    leftMargin=50, rightMargin=50,
    topMargin=50, bottomMargin=55,
    title="PMIS - Personal Memory Intelligence System Documentation",
    author="Yantra AI Labs",
    subject="Memory Structuring Rules & Component Reference"
)

doc.build(story, canvasmaker=NumberedCanvas)
print(f"PDF saved: {OUTPUT}")
