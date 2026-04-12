import streamlit as st
import plotly.express as px

def dashboard(df):

    st.subheader("📊 Анализ успеваемости")

    st.write("Распределение прогнозируемых баллов отражает общий уровень подготовки учащихся")

    # =========================
    # 📊 ГИСТОГРАММА
    # =========================
    fig = px.histogram(df, x="predicted_score")

    st.plotly_chart(fig)

    st.info("""
    📌 Интерпретация:
    - Левый хвост — учащиеся с низкой успеваемостью
    - Основная масса — средний уровень
    - Правый хвост — сильные учащиеся
    """)

    # =========================
    # 📦 СРАВНЕНИЕ ГРУПП
    # =========================
    st.subheader("📦 Сравнение групп учащихся")

    # --- средний балл ---
    grouped = df.groupby("student_type")["predicted_score"].mean().reset_index()

    fig2 = px.bar(
        grouped,
        x="student_type",
        y="predicted_score",
        text="predicted_score",
        title="Средний прогнозируемый балл по группам"
    )

    fig2.update_traces(texttemplate='%{text:.1f}', textposition='outside')

    st.plotly_chart(fig2)

    # --- размер групп ---
    group_size = df["student_type"].value_counts().reset_index()
    group_size.columns = ["student_type", "count"]

    fig_size = px.bar(
        group_size,
        x="student_type",
        y="count",
        text="count",
        title="Численность групп"
    )

    st.plotly_chart(fig_size)

    # --- разброс (оставляем boxplot) ---
    fig3 = px.box(
        df,
        x="student_type",
        y="predicted_score",
        title="Разброс успеваемости внутри групп"
    )

    st.plotly_chart(fig3)

    # =========================
    # 🧠 ТЕКСТОВАЯ АНАЛИТИКА
    # =========================
    st.subheader("🧠 Интерпретация результатов")

    best_group = grouped.sort_values("predicted_score", ascending=False).iloc[0]
    worst_group = grouped.sort_values("predicted_score").iloc[0]

    gap = best_group["predicted_score"] - worst_group["predicted_score"]

    st.markdown(f"""
    В результате анализа выявлены различия в уровне успеваемости между группами учащихся.

    📊 Наиболее высокая успеваемость наблюдается в группе **{best_group['student_type']}**
    (средний балл: **{best_group['predicted_score']:.1f}**).

    ⚠️ Наиболее низкие результаты демонстрирует группа **{worst_group['student_type']}**
    (средний балл: **{worst_group['predicted_score']:.1f}**).

    📉 Разрыв между группами составляет **{gap:.1f} баллов**, что свидетельствует
    о неоднородности образовательных результатов.

    👉 Это указывает на необходимость разработки адресных мер воздействия
    для повышения успеваемости учащихся с наихудшими показателями.
    """)

    # --- уровень проблемы ---
    if gap > 15:
        st.error("❗ Выявлен критический разрыв между группами — требуется срочное вмешательство")
    elif gap > 8:
        st.warning("⚠️ Наблюдается значительное различие между группами")
    else:
        st.success("✅ Различия между группами незначительные")

    st.caption("Анализ сформирован на основе модели машинного обучения и поведенческих факторов учащихся")