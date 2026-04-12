import pandas as pd
import numpy as np


# --- СИМУЛЯЦИЯ ИЗМЕНЕНИЙ ---
def simulate_change(row, feature):

    new_row = row.copy()

    if feature == "attendance":
        new_row[feature] = min(100, row[feature] + 20)

    elif feature == "homework_done":
        new_row[feature] = min(100, row[feature] + 20)

    elif feature == "motivation_level":
        new_row[feature] = min(1, row[feature] + 0.3)

    elif feature == "stress_level":
        new_row[feature] = max(0, row[feature] - 0.3)

    elif feature == "absences":
        new_row[feature] = max(0, row[feature] - 5)

    return new_row


# --- ИНТЕРПРЕТАЦИЯ ЭФФЕКТА ---
def interpret_gain(gain):
    if gain > 5:
        return "значительное улучшение"
    elif gain > 2:
        return "умеренное улучшение"
    elif gain > 0:
        return "незначительное улучшение"
    else:
        return None


# =========================
# 🧠 AI РЕКОМЕНДАЦИИ УЧЕНИКА
# =========================
def generate_student_ai(student_row, pipeline, features):

    student_df = pd.DataFrame([student_row])

    base_pred = pipeline.predict(student_df[features])[0]

    factors = ["attendance","homework_done","motivation_level","stress_level","absences"]

    insights = []

    for f in factors:

        try:
            new_row = simulate_change(student_row, f)

            new_pred = pipeline.predict(pd.DataFrame([new_row[features]]))[0]

            gain = new_pred - base_pred

            # --- фильтрация шума ---
            if abs(gain) < 0.5 or gain < 0:
                continue

            impact = interpret_gain(gain)

            if not impact:
                continue

            # --- генерация текста ---
            if f == "attendance":
                text = (
                    "Низкая посещаемость снижает академические результаты ученика. "
                    "Рекомендуется усиление контроля посещаемости и работа с причинами пропусков. "
                    f"Ожидается {impact} успеваемости."
                )

            elif f == "homework_done":
                text = (
                    "Недостаточное выполнение домашних заданий ухудшает закрепление материала. "
                    "Рекомендуется регулярный контроль и повышение дисциплины выполнения. "
                    f"Ожидается {impact} результатов."
                )

            elif f == "motivation_level":
                text = (
                    "Снижение мотивации уменьшает вовлечённость ученика в обучение. "
                    "Рекомендуется персонализация заданий и использование интерактивных методов. "
                    f"Ожидается {impact} успеваемости."
                )

            elif f == "stress_level":
                text = (
                    "Повышенный уровень стресса негативно влияет на когнитивные способности. "
                    "Рекомендуется снижение нагрузки и психологическая поддержка. "
                    f"Ожидается {impact} результатов."
                )

            elif f == "absences":
                text = (
                    "Частые пропуски занятий снижают общий уровень подготовки. "
                    "Рекомендуется выявление причин и профилактика пропусков. "
                    f"Ожидается {impact} успеваемости."
                )

            else:
                continue

            insights.append(text)

        except:
            continue

    # --- если нет инсайтов ---
    if not insights:
        insights.append(
            "Ученик демонстрирует стабильные показатели. "
            "Рекомендуется поддерживающая стратегия обучения."
        )

    return insights


# =========================
# 📊 ПРОГНОЗ УЧЕНИКА
# =========================
def generate_student_forecast(student_row, pipeline, features):

    student_df = pd.DataFrame([student_row])

    # текущий прогноз
    current_pred = pipeline.predict(student_df[features])[0]

    # --- ухудшение ---
    worse_row = student_row.copy()

    if "attendance" in worse_row:
        worse_row["attendance"] = max(0, worse_row["attendance"] - 10)

    if "motivation_level" in worse_row:
        worse_row["motivation_level"] = max(0, worse_row["motivation_level"] - 0.2)

    if "stress_level" in worse_row:
        worse_row["stress_level"] = min(1, worse_row["stress_level"] + 0.2)

    worse_pred = pipeline.predict(pd.DataFrame([worse_row[features]]))[0]

    # --- улучшение ---
    better_row = student_row.copy()

    if "attendance" in better_row:
        better_row["attendance"] = min(100, better_row["attendance"] + 15)

    if "homework_done" in better_row:
        better_row["homework_done"] = min(100, better_row["homework_done"] + 15)

    if "motivation_level" in better_row:
        better_row["motivation_level"] = min(1, better_row["motivation_level"] + 0.2)

    if "stress_level" in better_row:
        better_row["stress_level"] = max(0, better_row["stress_level"] - 0.2)

    better_pred = pipeline.predict(pd.DataFrame([better_row[features]]))[0]

    return {
        "current": current_pred,
        "worse": worse_pred,
        "better": better_pred
    }