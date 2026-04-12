import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# =========================
# 🎯 ОБУЧЕНИЕ МОДЕЛИ РИСКА
# =========================
def train_risk_model(df, features):

    NUMERIC_FEATURES = [
        "attendance","homework_done","avg_grade","discipline_score",
        "engagement_level","absences","teacher_rating","stress_level",
        "motivation_level","extra_classes"
    ]

    df = df.copy()

    # --- АДАПТИВНЫЙ TARGET ---
    threshold = df["avg_grade"].median()
    df["risk_target"] = (df["avg_grade"] < threshold).astype(int)

    X = df[NUMERIC_FEATURES]
    y = df["risk_target"]

    # защита от одного класса
    if len(np.unique(y)) < 2:
        y = (df["avg_grade"] < df["avg_grade"].mean()).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X_scaled, y)

    return model, scaler


# =========================
# 📊 РАСЧЁТ РИСКА
# =========================
def calculate_risk(df, model, scaler, features):

    NUMERIC_FEATURES = [
        "attendance","homework_done","avg_grade","discipline_score",
        "engagement_level","absences","teacher_rating","stress_level",
        "motivation_level","extra_classes"
    ]

    df = df.copy()

    X = df[NUMERIC_FEATURES]
    X_scaled = scaler.transform(X)

    # --- защита predict_proba ---
    proba = model.predict_proba(X_scaled)

    if proba.shape[1] == 1:
        risk_proba = np.zeros(len(df))
    else:
        risk_proba = proba[:, 1]

    # --- сглаживание ---
    risk_proba = np.clip(risk_proba, 0, 1)

    df["risk_score"] = risk_proba
    df["risk_level"] = df["risk_score"].apply(get_risk_level)

    return df


# =========================
# 🚦 УРОВЕНЬ РИСКА
# =========================
def get_risk_level(score):
    if score > 0.75:
        return "🔴 высокий"
    elif score > 0.45:
        return "🟡 средний"
    else:
        return "🟢 низкий"


# =========================
# 🔍 ОБЪЯСНЕНИЕ РИСКА
# =========================
def explain_risk(row, df):

    reasons = []
    means = df.mean(numeric_only=True)

    if row["attendance"] < means["attendance"]:
        reasons.append("Посещаемость ниже среднего")

    if row["homework_done"] < means["homework_done"]:
        reasons.append("Слабое выполнение домашних заданий")

    if row["motivation_level"] < means["motivation_level"]:
        reasons.append("Низкая учебная мотивация")

    if row["stress_level"] > means["stress_level"]:
        reasons.append("Повышенный уровень стресса")

    if row["absences"] > means["absences"]:
        reasons.append("Частые пропуски занятий")

    return reasons


# =========================
# 🧠 РЕКОМЕНДАЦИИ (УЛУЧШЕННЫЕ)
# =========================
def generate_risk_recommendations(row):

    recs = []

    if row["attendance"] < 60:
        recs.append("🔴 Критично: повысить посещаемость (работа с родителями, контроль)")

    if row["homework_done"] < 50:
        recs.append("🟠 Усилить контроль выполнения домашних заданий")

    if row["motivation_level"] < 0.4:
        recs.append("🟠 Повысить мотивацию через персонализацию и интерактив")

    if row["stress_level"] > 0.7:
        recs.append("🔴 Снизить стресс: уменьшить нагрузку, подключить психолога")

    if row["absences"] > 10:
        recs.append("🟠 Проанализировать причины пропусков")

    if not recs:
        recs.append("🟢 Риск низкий — поддерживать текущую стратегию")

    return recs


# =========================
# 📊 ВАЖНОСТЬ ФАКТОРОВ
# =========================
def get_risk_feature_importance(model):

    features = [
        "attendance","homework_done","avg_grade","discipline_score",
        "engagement_level","absences","teacher_rating","stress_level",
        "motivation_level","extra_classes"
    ]

    importances = model.feature_importances_

    return pd.DataFrame({
        "feature": features,
        "importance": importances
    }).sort_values("importance", ascending=False)


# =========================
# 📊 АНАЛИТИКА РИСКА
# =========================
def generate_risk_insights(df):

    insights = []

    high = df[df["risk_score"] > 0.75]

    if len(high) > 0:

        share = len(high) / len(df) * 100

        insights.append(f"Доля учащихся высокого риска: {share:.1f}%")
        insights.append(f"Средняя посещаемость: {high['attendance'].mean():.1f}")
        insights.append(f"Средняя мотивация: {high['motivation_level'].mean():.2f}")
        insights.append(f"Средний стресс: {high['stress_level'].mean():.2f}")

    else:
        insights.append("Группа высокого риска не выявлена")

    return insights


# =========================
# 🚀 ПРИОРИТЕТ УЧЕНИКОВ (NEW)
# =========================
def get_risk_priority(df):

    df = df.copy()

    df["priority_score"] = (
        df["risk_score"] * 0.6 +
        (1 - df["attendance"] / 100) * 0.2 +
        df["stress_level"] * 0.2
    )

    return df.sort_values("priority_score", ascending=False)