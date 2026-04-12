import numpy as np

FEATURES = [
    "attendance","homework_done","avg_grade",
    "engagement_level","absences","stress_level","motivation_level"
]

def generate_ai_recommendations(df):

    means = df[FEATURES].mean()

    recs = []
    priorities = []

    for _, row in df.iterrows():

        score = 0
        text = []

        # --- логика ---
        if row["attendance"] < means["attendance"]:
            text.append("Повысить посещаемость")
            score += 2

        if row["homework_done"] < means["homework_done"]:
            text.append("Усилить контроль ДЗ")
            score += 2

        if row["stress_level"] > means["stress_level"]:
            text.append("Снизить стресс")
            score += 1

        if row["motivation_level"] < means["motivation_level"]:
            text.append("Повысить мотивацию")
            score += 2

        if row["engagement_level"] < means["engagement_level"]:
            text.append("Повысить вовлеченность")
            score += 2

        if not text:
            text = ["Поддерживать текущую стратегию"]

        # --- приоритет ---
        if score >= 6:
            pr = "Высокий"
        elif score >= 3:
            pr = "Средний"
        else:
            pr = "Низкий"

        recs.append("; ".join(text))
        priorities.append(pr)

    df["recommendation"] = recs
    df["priority"] = priorities

    return df