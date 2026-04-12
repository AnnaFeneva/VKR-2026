import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
import textwrap

def print_block(pdf, text, line_height=6, max_width=80, indent=0):
    lines = textwrap.wrap(str(text), width=max_width, break_long_words=True)
    
    for line in lines:
        if indent:
            pdf.cell(indent)
        pdf.cell(0, line_height, line, ln=True)

# --- безопасный текст ---
def safe_text(text, width=90):
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=True))


# --- анализ класса ---
def analyze_class(df):

    metrics = {
        "avg_grade": df["avg_grade"].mean(),
        "attendance": df["attendance"].mean(),
        "motivation": df["motivation_level"].mean(),
        "stress": df["stress_level"].mean()
    }

    problems = []
    strengths = []

    if metrics["attendance"] < 60:
        problems.append("Низкая посещаемость")
    else:
        strengths.append("Хорошая посещаемость")

    if metrics["motivation"] < 0.5:
        problems.append("Низкая мотивация")
    else:
        strengths.append("Хорошая мотивация")

    if metrics["stress"] > 0.6:
        problems.append("Высокий стресс")
    else:
        strengths.append("Контролируемый уровень стресса")

    if metrics["avg_grade"] < 60:
        problems.append("Слабая успеваемость")
    else:
        strengths.append("Хорошая успеваемость")

    return metrics, problems, strengths


def generate_class_report(df, result, title="Аналитический отчёт"):

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.add_font("DejaVu", "B", "DejaVuSans-Bold.ttf", uni=True)

    page_width = pdf.w - 2 * pdf.l_margin

    classes = df["class"].unique()

    for cls in classes:
        df_class = df[df["class"] == cls]

        metrics, problems, strengths = analyze_class(df_class)

        pdf.add_page()

        # ---------- Заголовок ----------
        pdf.set_font("DejaVu", 'B', 16)
        pdf.cell(0, 10, f"{title} — Класс {cls}", ln=True, align='C')
        pdf.ln(5)

        # ---------- СВОДКА ----------
        pdf.set_font("DejaVu", 'B', 12)
        pdf.cell(0, 8, "Краткая сводка:", ln=True)

        pdf.set_font("DejaVu", '', 11)

        summary = (
            f"Средний балл: {metrics['avg_grade']:.1f}, "
            f"Посещаемость: {metrics['attendance']:.1f}%, "
            f"Мотивация: {metrics['motivation']:.2f}, "
            f"Стресс: {metrics['stress']:.2f}"
        )

        pdf.multi_cell(page_width, 6, safe_text(summary))
        pdf.ln(3)

        # ---------- ПРОБЛЕМЫ ----------
        pdf.set_font("DejaVu", 'B', 12)
        pdf.cell(0, 8, "Основные проблемы:", ln=True)

        pdf.set_font("DejaVu", '', 11)
        for p in problems[:3]:
            pdf.multi_cell(page_width, 6, f"🔴 {p}")

        pdf.ln(3)

        # ---------- СИЛЬНЫЕ СТОРОНЫ ----------
        pdf.set_font("DejaVu", 'B', 12)
        pdf.cell(0, 8, "Сильные стороны:", ln=True)

        pdf.set_font("DejaVu", '', 11)
        for s in strengths[:3]:
            print_block(pdf,f"🟢 {s}", indent=5)

        pdf.ln(5)

        # ---------- AI ВЫВОД ----------
        pdf.set_font("DejaVu", 'B', 12)
        pdf.cell(0, 8, "AI-заключение:", ln=True)

        pdf.set_font("DejaVu", '', 11)

        ai_text = result.get("report", "")
        pdf.multi_cell(page_width, 6, safe_text(ai_text))
        pdf.ln(5)

        # ---------- УЧЕНИКИ ----------
        pdf.set_font("DejaVu", 'B', 12)
        pdf.cell(0, 8, "Ученики:", ln=True)

        pdf.set_font("DejaVu", '', 10)

        import textwrap

        
        max_rows = 45
        df_display = df_class.head(max_rows)

        for _, row in df_display.iterrows():
            text = (
                f"{str(row['name'])[:20]} | "
                f"балл={row['avg_grade']:.1f} | "
                f"посещ={row['attendance']:.0f}% | "
                f"мотив={row['motivation_level']:.2f} | "
                f"стресс={row['stress_level']:.2f}"
            )

            # перенос строки вручную
            lines = textwrap.wrap(text, width=80)

            for line in lines:
                pdf.cell(0, 6, line, ln=True)

        pdf.ln(5)

        # ---------- РЕКОМЕНДАЦИИ ----------
        pdf.set_font("DejaVu", 'B', 12)
        pdf.cell(0, 8, "Рекомендации:", ln=True)

        pdf.set_font("DejaVu", '', 11)
        for r in result.get("recommendations", []):
            print_block(pdf,f"• {safe_text(r)}", indent=5)

        pdf.ln(5)

        # ---------- ГРАФИК ----------
        fig, ax = plt.subplots(figsize=(6,4))

        ax.bar(
            ["Успеваемость", "Посещаемость", "Мотивация", "Стресс"],
            [
                metrics["avg_grade"],
                metrics["attendance"],
                metrics["motivation"] * 100,
                metrics["stress"] * 100
            ]
        )

        ax.set_title(f"Класс {cls}")
        ax.set_ylim(0, 100)

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        # ❗ ВАЖНО: новая страница
        pdf.add_page()

        pdf.set_font("DejaVu", 'B', 12)
        pdf.cell(0, 10, f"Графики — класс {cls}", ln=True)

        pdf.image(buf, x=10, y=30, w=180)

    pdf_bytes = pdf.output(dest='S')
    return BytesIO(pdf_bytes)