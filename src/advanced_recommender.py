import numpy as np

NUMERIC_FEATURES = [
    "attendance","homework_done","avg_grade","discipline_score",
    "engagement_level","absences","teacher_rating","stress_level",
    "motivation_level","extra_classes"
]

def generate_smart_recommendations(df, model):
    df = df.copy()

    means = df[NUMERIC_FEATURES].mean()

    recs = []
    reasons = []
    impact_scores = []

    for _, row in df.iterrows():
        student_recs = []
        student_reasons = []
        impact = 0

        # 🔍 Анализ отклонений от нормы
        for f in NUMERIC_FEATURES:
            diff = row[f] - means[f]

            # Негативное отклонение
            if diff < -0.2 * means[f]:
                impact += abs(diff)

                if f == "attendance":
                    student_recs.append("Усилить контроль посещаемости")
                    student_reasons.append("Низкая посещаемость снижает итоговый балл")

                elif f == "homework_done":
                    student_recs.append("Увеличить контроль выполнения ДЗ")
                    student_reasons.append("Недостаточная работа дома")

                elif f == "motivation_level":
                    student_recs.append("Повысить мотивацию через персональные цели")
                    student_reasons.append("Низкая мотивация влияет на результат")

                elif f == "stress_level":
                    student_recs.append("Снизить учебную нагрузку")
                    student_reasons.append("Высокий стресс ухудшает успеваемость")

                elif f == "engagement_level":
                    student_recs.append("Повысить вовлеченность (интерактив, проекты)")
                    student_reasons.append("Низкая вовлеченность")

        if not student_recs:
            student_recs = ["Поддерживать текущую стратегию обучения"]
            student_reasons = ["Показатели ученика стабильны"]

        # 🎯 Приоритет
        if impact > 50:
            priority = "Высокий"
        elif impact > 20:
            priority = "Средний"
        else:
            priority = "Низкий"

        recs.append("; ".join(set(student_recs)))
        reasons.append("; ".join(set(student_reasons)))
        impact_scores.append(priority)

    df["recommendation"] = recs
    df["reason"] = reasons
    df["priority"] = impact_scores

    return df