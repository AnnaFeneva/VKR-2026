import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.student_ai import generate_student_ai, generate_student_forecast


REQUIRED_COLUMNS = [
    "class","school_type","parent_education","family_income","internet_access",
    "attendance","homework_done","avg_grade","discipline_score",
    "engagement_level","absences","teacher_rating","stress_level",
    "motivation_level","extra_classes"
]


NUM_COLS = [
    "attendance","homework_done","avg_grade","discipline_score",
    "engagement_level","absences","teacher_rating","stress_level",
    "motivation_level","extra_classes"
]


def show_student_card(df, pipeline, features):

    selected = st.selectbox("Выберите ученика", df["name"].unique())
    student = df[df["name"] == selected].iloc[0]

    X_student = pd.DataFrame([student[REQUIRED_COLUMNS]])

    # --- прогноз ---
    pred = pipeline.predict(X_student)[0]
    st.metric("📊 Прогноз балла", f"{pred:.2f}")

    # --- влияние факторов ---
    st.subheader("📈 Влияние факторов")

    diffs = {}
    means = df[NUM_COLS].mean()

    for col in NUM_COLS:
        diffs[col] = student[col] - means[col]

    impact_df = pd.DataFrame({
        "feature": list(diffs.keys()),
        "impact": list(diffs.values())
    }).sort_values("impact")

    st.bar_chart(impact_df.set_index("feature"))

    # --- симуляция ---
    st.subheader("🧪 Что если улучшить ученика")

    col1, col2 = st.columns(2)

    with col1:
        attendance = st.slider("Посещаемость", 0, 100, int(student["attendance"]), 1)
        homework = st.slider("Домашка", 0, 100, int(student["homework_done"]), 1)

    with col2:
        motivation = st.slider("Мотивация", 0.0, 1.0, float(student["motivation_level"]), 0.01)
        stress = st.slider("Стресс", 0.0, 1.0, float(student["stress_level"]), 0.01)

    if st.button("🚀 Пересчитать результат"):

        X_new = X_student.copy()

        X_new["attendance"] = attendance
        X_new["homework_done"] = homework
        X_new["motivation_level"] = motivation
        X_new["stress_level"] = stress

        new_pred = pipeline.predict(X_new)[0]

        st.success(f"Новый прогноз: {new_pred:.2f}")

    # =========================
    # 🧠 AI АНАЛИЗ УЧЕНИКА
    # =========================

    st.divider()
    st.subheader("🧠 AI-рекомендации")

    ai_recs = generate_student_ai(student, pipeline, features)

    for rec in ai_recs:
        st.write("•", rec)

    # =========================
    # 📊 ПРОГНОЗ
    # =========================

    st.divider()
    st.subheader("📊 Прогноз успеваемости")

    forecast = generate_student_forecast(student, pipeline, features)

    current = forecast["current"]
    worse = forecast["worse"]
    better = forecast["better"]

    delta_worse = worse - current
    delta_better = better - current

    # --- метрики ---
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Без вмешательства", f"{worse:.1f}", f"{delta_worse:.1f}")

    with col2:
        st.metric("С рекомендациями", f"{better:.1f}", f"+{delta_better:.1f}")

    # --- уровень ---
    def get_level(score):
        if score >= 80:
            return "🟢 высокий"
        elif score >= 60:
            return "🟡 средний"
        else:
            return "🔴 низкий"

    st.write(f"Текущий уровень: {get_level(current)}")

    # --- прогресс ---
    st.write("📈 Текущий прогресс")
    st.progress(min(max(current / 100, 0), 1))

    # --- график ---
    st.subheader("📉 Динамика сценариев")

    values = [worse, current, better]
    labels = ["Без вмешательства", "Текущий", "С рекомендациями"]

    fig, ax = plt.subplots()
    ax.plot(labels, values, marker='o')
    ax.set_title("Прогноз изменения успеваемости")
    ax.set_ylabel("Баллы")

    st.pyplot(fig)