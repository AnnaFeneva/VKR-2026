def generate_insights(df):

    insights = []

    if df["attendance"].mean() < 60:
        insights.append("⚠️ Низкая посещаемость")

    if df["stress_level"].mean() > 0.6:
        insights.append("⚠️ Высокий уровень стресса")

    if df["motivation_level"].mean() < 0.5:
        insights.append("⚠️ Низкая мотивация учащихся")

    if not insights:
        insights.append("✅ Система работает эффективно")

    return insights