"""NeurIPS-style figure: Application | Generator–Verifier interaction | SAC.

All shapes are native PowerPoint primitives so the slide is fully editable.
Aesthetic spec:
  • 2 muted accents (teal, rust) + 3 greys
  • flat rectangles, 1-pt borders, no shadows / gradients / heavy rounding
  • mono for code, sans for labels, bold only for headers and the verdict
  • thin (0.75–1 pt) consistent stroke for arrows
  • generous whitespace, equal-height panels aligned to a baseline grid

Output: filer_verifier_sac.pptx
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

OUT_DIR = Path(__file__).resolve().parent
OUT_PATH = OUT_DIR / "filer_verifier_sac.pptx"

# ---------- Palette ----------
TEAL = RGBColor(0x2C, 0x5F, 0x73)
RUST = RGBColor(0xB4, 0x5F, 0x33)
INK = RGBColor(0x2A, 0x2A, 0x2A)
GREY = RGBColor(0x6A, 0x6A, 0x6A)
LINE = RGBColor(0xC8, 0xC8, 0xC8)
PANEL = RGBColor(0xFA, 0xFA, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SANS = "Calibri"
SERIF = "Cambria"
MONO = "Consolas"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


# ---------- Helpers ----------
def add_textbox(slide, x, y, w, h, runs, *, align=PP_ALIGN.LEFT, fill=None, border=None):
    """Insert a text box. ``runs`` is a list of paragraph-tuples:
    [(text, font, size_pt, bold, italic, color), ...]; a None text starts a new paragraph.
    """
    if fill is not None or border is not None:
        rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
        if fill is not None:
            rect.fill.solid()
            rect.fill.fore_color.rgb = fill
        else:
            rect.fill.background()
        if border is None:
            rect.line.fill.background()
        else:
            rect.line.color.rgb = border
            rect.line.width = Pt(0.75)
        tf = rect.text_frame
    else:
        box = slide.shapes.add_textbox(x, y, w, h)
        tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.14)
    tf.margin_right = Inches(0.14)
    tf.margin_top = Inches(0.10)
    tf.margin_bottom = Inches(0.10)
    first = True
    for item in runs:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.alignment = align
        text, font, size, bold, italic, color = item
        if text is None:
            continue
        run = p.add_run()
        run.text = text
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color


def add_panel(slide, x, y, w, h, header_text, header_color, blocks):
    """Header bar (accent color) + stack of light-fill blocks.
    ``blocks`` = [(label, body_runs, height_in)].
    """
    # Header
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, Inches(0.42))
    bar.fill.solid()
    bar.fill.fore_color.rgb = header_color
    bar.line.fill.background()
    add_textbox(
        slide, x, y, w, Inches(0.42),
        [(header_text, SANS, 16, True, False, WHITE)],
    )
    # Blocks
    cy = y + Inches(0.50)
    gap = Inches(0.12)
    for label, body_runs, h_in in blocks:
        bh = Inches(h_in)
        runs = [(label, SANS, 11, True, False, header_color)]
        runs.extend(body_runs)
        add_textbox(
            slide, x, cy, w, bh, runs,
            fill=PANEL, border=LINE,
        )
        cy += bh + gap


def code_runs(lines, *, size=10):
    """Return list of paragraph-tuples for a monospace JSON/code snippet."""
    out = [(None, MONO, size, False, False, INK)]  # spacer paragraph after label
    for ln in lines:
        out.append((ln if ln else " ", MONO, size, False, False, INK))
    return out


def text_runs(lines, *, size=11, italic=False):
    out = [(None, SANS, size, False, False, INK)]
    for ln in lines:
        out.append((ln if ln else " ", SANS, size, False, italic, INK))
    return out


def add_arrow(slide, x1, y1, x2, y2, *, color=INK, width=Pt(1.0)):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    line.line.color.rgb = color
    line.line.width = width
    # Arrowhead at the end
    el = line.line._get_or_add_ln()
    from pptx.oxml.ns import qn
    from lxml import etree
    tail = etree.SubElement(el, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("h", "med")
    return line


def add_label(slide, x, y, w, h, text, *, size=10, italic=True, color=INK, align=PP_ALIGN.CENTER):
    add_textbox(
        slide, x, y, w, h,
        [(text, SERIF, size, False, italic, color)],
        align=align,
    )


def add_agent_box(slide, x, y, w, h, name, math_label, color):
    rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    rect.fill.solid()
    rect.fill.fore_color.rgb = WHITE
    rect.line.color.rgb = color
    rect.line.width = Pt(1.5)
    add_textbox(
        slide, x, y, w, h,
        [
            (name, SANS, 14, True, False, color),
            (math_label, SERIF, 16, False, True, INK),
        ],
        align=PP_ALIGN.CENTER,
    )


def build_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

    # Background fill = white (default)
    # ---- Title ----
    add_textbox(
        slide, Inches(0.5), Inches(0.20), Inches(12.3), Inches(0.55),
        [("Filer–Verifier line-level interaction with safety cases",
          SANS, 24, True, False, INK)],
    )
    add_textbox(
        slide, Inches(0.5), Inches(0.78), Inches(12.3), Inches(0.30),
        [("Real case excerpt from TaxCalcBench (ty24).",
          SERIF, 12, False, True, GREY)],
    )

    # ---------- Layout grid ----------
    margin_x = Inches(0.5)
    panel_top = Inches(1.30)
    panel_h = Inches(5.55)
    left_w = Inches(3.6)
    right_w = Inches(3.6)
    center_x = margin_x + left_w + Inches(0.20)
    center_w = SLIDE_W - 2 * margin_x - left_w - right_w - Inches(0.40)
    right_x = SLIDE_W - margin_x - right_w

    # ---------- LEFT: Application ----------
    add_panel(
        slide, margin_x, panel_top, left_w, panel_h,
        "Application", TEAL,
        [
            (
                "Tax-form text  (excerpt)",
                code_runs([
                    "Form 1040  →  Schedule 1",
                    "Line 21:  Student loan interest",
                    "          deduction",
                    "          (IRC §221(b), max $2,500)",
                ], size=10),
                1.30,
            ),
            (
                "Input data  (input.json, abridged)",
                code_runs([
                    "filing_status        : MFJ",
                    "W-2 wages (Box 1)    : $15,111",
                    "Schedule C net      : $23,210",
                    "Student loan int.    : $3,000  ← 1098-E",
                    "AGI (recomputed)     : $36,681",
                ], size=10),
                1.55,
            ),
            (
                "Filer prompt  (line-level)",
                code_runs([
                    "System: output ONLY valid JSON",
                    "        for the target line.",
                    "User:",
                    "  input         : <input.json>",
                    "  target_line   : Sched. 1 / 21",
                    "  context_so_far: [...prior lines...]",
                    "Return: { form, line, amount,",
                    "          rationale }",
                ], size=10),
                1.85,
            ),
        ],
    )

    # ---------- CENTER: Generator–Verifier interaction ----------
    # Two agent boxes at the top of the center column.
    agent_w = Inches(2.0)
    agent_h = Inches(0.95)
    agent_top = panel_top + Inches(0.20)
    cx_left = center_x + Inches(0.10)
    cx_right = center_x + center_w - Inches(0.10) - agent_w

    add_agent_box(slide, cx_left, agent_top, agent_w, agent_h,
                  "Filer", "G ᵍ", TEAL)
    add_agent_box(slide, cx_right, agent_top, agent_w, agent_h,
                  "Verifier", "V ᵛ", RUST)

    # Two horizontal arrows between them
    a_y_top = agent_top + Inches(0.25)
    a_y_bot = agent_top + Inches(0.70)
    add_arrow(slide, cx_left + agent_w, a_y_top, cx_right, a_y_top,
              color=TEAL, width=Pt(1.25))
    add_arrow(slide, cx_right, a_y_bot, cx_left + agent_w, a_y_bot,
              color=RUST, width=Pt(1.25))
    add_label(slide, cx_left + agent_w, a_y_top - Inches(0.32),
              cx_right - (cx_left + agent_w), Inches(0.30),
              "draft  xₗ", size=11, color=TEAL)
    add_label(slide, cx_left + agent_w, a_y_bot + Inches(0.04),
              cx_right - (cx_left + agent_w), Inches(0.30),
              "safety case  Sₗ = (cₗ, gₗ, eₗ)", size=11, color=RUST)

    # Decision box below the filer
    dec_w = Inches(2.7)
    dec_h = Inches(0.75)
    dec_x = center_x + (center_w - dec_w) / 2
    dec_y = agent_top + agent_h + Inches(0.85)
    dec = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, dec_x, dec_y, dec_w, dec_h)
    dec.fill.solid()
    dec.fill.fore_color.rgb = PANEL
    dec.line.color.rgb = LINE
    dec.line.width = Pt(0.75)
    add_textbox(
        slide, dec_x, dec_y, dec_w, dec_h,
        [
            ("Filer decision", SANS, 11, True, False, INK),
            ("KEEP   xₗ        |        REVISE → zₗ", SERIF, 13, False, True, INK),
        ],
        align=PP_ALIGN.CENTER,
    )

    # Arrow from filer down through midpoint then into decision
    add_arrow(slide, cx_left + agent_w / 2, agent_top + agent_h,
              cx_left + agent_w / 2, dec_y, width=Pt(1.0))
    add_label(slide,
              cx_left + agent_w / 2 + Inches(0.05),
              agent_top + agent_h + Inches(0.05),
              Inches(1.5), Inches(0.30),
              "uses Sₗ", size=10, italic=True, color=GREY,
              align=PP_ALIGN.LEFT)

    # Reward boxes side by side, below decision
    rw_w = Inches(1.85)
    rw_h = Inches(0.70)
    rw_y = dec_y + dec_h + Inches(0.55)
    rw_left_x = center_x + (center_w / 2) - rw_w - Inches(0.10)
    rw_right_x = center_x + (center_w / 2) + Inches(0.10)

    # Filer reward
    rb1 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, rw_left_x, rw_y, rw_w, rw_h)
    rb1.fill.solid(); rb1.fill.fore_color.rgb = WHITE
    rb1.line.color.rgb = TEAL; rb1.line.width = Pt(1.25)
    add_textbox(
        slide, rw_left_x, rw_y, rw_w, rw_h,
        [
            ("Filer reward", SANS, 11, True, False, TEAL),
            ("Rᵍ = Sₓ  (or Sᵤ)", SERIF, 12, False, True, INK),
        ],
        align=PP_ALIGN.CENTER,
    )
    # Verifier reward
    rb2 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, rw_right_x, rw_y, rw_w, rw_h)
    rb2.fill.solid(); rb2.fill.fore_color.rgb = WHITE
    rb2.line.color.rgb = RUST; rb2.line.width = Pt(1.25)
    add_textbox(
        slide, rw_right_x, rw_y, rw_w, rw_h,
        [
            ("Verifier reward", SANS, 11, True, False, RUST),
            ("Rᵛ = 𝟙[SAC ↔ wrong]", SERIF, 12, False, True, INK),
        ],
        align=PP_ALIGN.CENTER,
    )
    # Arrows from decision down to each reward
    add_arrow(slide, dec_x + dec_w * 0.30, dec_y + dec_h,
              rw_left_x + rw_w / 2, rw_y, width=Pt(1.0))
    add_arrow(slide, dec_x + dec_w * 0.70, dec_y + dec_h,
              rw_right_x + rw_w / 2, rw_y, width=Pt(1.0))

    # Caption beneath the central diagram
    cap_y = rw_y + rw_h + Inches(0.20)
    add_textbox(
        slide, center_x, cap_y, center_w, Inches(0.60),
        [
            ("One line ℓ — three transitions in a single rollout: "
             "filer drafts xₗ, verifier issues SAC Sₗ or NO_SAC, "
             "filer either keeps or revises. Both agents receive paired-action GRPO rewards.",
             SERIF, 10, False, True, GREY),
        ],
        align=PP_ALIGN.LEFT,
    )

    # ---------- RIGHT: SAC ----------
    add_panel(
        slide, right_x, panel_top, right_w, panel_h,
        "SAC", RUST,
        [
            (
                "Generator draft  xₗ",
                code_runs([
                    "{ \"form\": \"Schedule 1\",",
                    "  \"line\": \"21\",",
                    "  \"amount\": 3000,",
                    "  \"rationale\":",
                    "    \"1098-E reports $3,000 of",
                    "     student loan interest paid.\" }",
                ], size=10),
                1.55,
            ),
            (
                "cₗ   claim",
                text_runs(["“Line 21 exceeds its statutory cap.”"], size=11, italic=True),
                0.65,
            ),
            (
                "gₗ   grounding rule",
                text_runs([
                    "“Deduction = min(amount paid, $2,500);",
                    " the draft uses the amount paid only.”",
                ], size=11, italic=True),
                0.95,
            ),
            (
                "eₗ   numerical evidence",
                text_runs([
                    "Paid (1098-E) = $3,000",
                    "Cap (IRC §221(b)) = $2,500",
                    "Expected = min($3,000, $2,500) = $2,500",
                ], size=11),
                1.10,
            ),
            (
                "Verdict",
                text_runs([
                    "WRONG  →  SAC fires;  suggested fix: $2,500",
                ], size=12),
                0.55,
            ),
        ],
    )


def main() -> None:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    build_slide(prs)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
