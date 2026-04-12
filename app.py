import streamlit as st
import pandas as pd
import joblib

# ML
from src.ml_pipeline import run_pipeline
from src.student_card import show_student_card
from src.visualization import dashboard
from src.insights import generate_insights
from src.clustering import (
    show_clusters,
    show_cluster_heatmap,
    generate_cluster_insights,
    generate_class_recommendations
)

# Backend
from backend.routes import get_logs, authenticate_user, create_user, get_all_users, delete_user



# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Edu AI System",
    layout="wide",
    page_icon="🎓"
)

# ---------------- SESSION ----------------
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("user", None)

# ---------------- LOGOUT ----------------
def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()

# ---------------- LOGIN ----------------
def login_page():
    st.title("🎓 Edu AI System")
    st.subheader("🔐 Вход")

    login = st.text_input("Логин")
    password = st.text_input("Пароль", type="password")

    if st.button("Войти"):
        user = authenticate_user(login, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.user = user
            st.rerun()
        else:
            st.error("❌ Неверный логин")

# ---------------- LOAD ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/students_realistic_big.csv")

@st.cache_resource
def load_pipeline():
    return joblib.load("models/pipeline.pkl")

@st.cache_data
def run_cached_pipeline(df):
    return run_pipeline(df)

# ---------------- MAIN ----------------
def main_app():

    user = st.session_state.user

    # Sidebar
    st.sidebar.title("🎓 Edu AI System")
    st.sidebar.write(f"👤 {user['login']} ({user['role']})")
    st.sidebar.button("🚪 Выйти", on_click=logout)

    menu = st.sidebar.radio("Навигация", [
        "Дашборд",
        "Карточка ученика",
        "Рекомендации",
        "Риск",
        "Кластеры",
        # "Инсайты",
        "Отчет",
        "Пользователи",
        "Логи",
        "Explainability"
    ])

    # Data
    df = load_data()
    pipeline = load_pipeline()

    # --- Фильтр по классу (для админа) ---
    if user.get("role") == "admin":

        st.sidebar.subheader("🔎 Фильтр данных")

        classes = sorted(df["class"].dropna().unique().tolist())

        selected_class = st.sidebar.selectbox(
            "Выберите класс",
            ["Все"] + classes
        )

        if selected_class != "Все":
            df = df[df["class"] == selected_class]
            st.sidebar.success(f"Фильтр: {selected_class}")
    
    
    st.sidebar.markdown("---")
    st.sidebar.write("📊 Статистика")

    st.sidebar.write(f"👥 Учеников: {len(df)}")

    if "risk_label" in df.columns:
        high_risk = (df["risk_label"] == "высокий").sum()
        st.sidebar.write(f"🔴 Риск: {high_risk}")

    # Фильтр учителя
    if user.get("role") == "teacher" and user.get("class"):
        df = df[df["class"] == user["class"]]
        st.sidebar.success(f"Фильтр: {user['class']}")

    # ML pipeline
    with st.spinner("⏳ Работа AI системы..."):
        df, _, features = run_cached_pipeline(df)

    # ---------------- PAGES ----------------

    if menu == "Дашборд":
        st.title("📊 Дашборд")
        dashboard(df)

    elif menu == "Карточка ученика":
        st.title("👩‍🎓 Карточка ученика")
        show_student_card(df, pipeline, features)

    elif menu == "Рекомендации":

        from src.ai_strategy import generate_ai_strategy

        st.title("🧠 AI рекомендации")

        result = generate_ai_strategy(df, pipeline, features)

        # --- график ---
        import plotly.express as px

        fig = px.bar(
            result["importance"].head(7),
            x="importance",
            y="feature",
            orientation="h",
            title="Факторы влияния на успеваемость"
        )

        fig.update_layout(yaxis=dict(autorange="reversed"))

        st.plotly_chart(fig, use_container_width=True)

        # --- текст ---
        st.subheader("📊 Анализ")

        st.info(result["report"])

        # --- рекомендации ---
        st.subheader("🚀 Рекомендации")

        for r in result["recommendations"]:
            st.success(r)

    elif menu == "Риск":

        from src.risk_model import (
            explain_risk,
            generate_risk_recommendations,
            generate_risk_insights,
            get_risk_priority
        )

        st.title("⚠️ Анализ риска")

        st.info("""
        📌 Risk Score рассчитывается на основе:
        - посещаемости
        - количества пропусков
        - уровня стресса
        - мотивации

        Чем выше значение — тем выше риск снижения успеваемости.
        """)

        st.divider()

        df_risk = df.copy()

        # =========================
        # 📊 ОСНОВНАЯ ТАБЛИЦА (СОРТИРОВКА + ТОП)
        # =========================

        st.subheader("📊 Ученики по уровню риска")

        df_sorted = df_risk.sort_values("risk_score", ascending=False).copy()

        df_sorted["top_risk"] = df_sorted["risk_score"].apply(
            lambda x: "🔥 ТОП" if x > 0.75 else ""
        )

        st.dataframe(
            df_sorted[["name", "risk_score", "risk_level", "top_risk"]],
            use_container_width=True
        )

        st.divider()

        # =========================
        # 🔍 ПРИЧИНЫ РИСКА (НЕ ПРОСТЫНЯ)
        # =========================

        st.subheader("🔍 Причины риска")

        mode = st.radio(
            "Режим отображения",
            ["Топ 10", "Только риск", "Все"],
            horizontal=True
        )

        df_view = df_sorted.copy()

        if mode == "Топ 10":
            df_view = df_view.head(10)

        elif mode == "Только риск":
            df_view = df_view[df_view["risk_score"] > 0.4]

        for _, row in df_view.iterrows():

            if row["risk_score"] > 0.75:
                icon = "🔴"
            elif row["risk_score"] > 0.45:
                icon = "🟡"
            else:
                icon = "🟢"

            with st.expander(f"{icon} {row['name']} ({row['risk_level']})"):

                reasons = explain_risk(row, df_risk)

                if reasons:
                    for r in reasons:
                        st.write("•", r)
                else:
                    st.write("Нет выраженных факторов риска")

        st.divider()

        # =========================
        # 🧠 РЕКОМЕНДАЦИИ (ТОЖЕ УМНО)
        # =========================

        st.subheader("🧠 Рекомендации")

        for _, row in df_view.iterrows():

            with st.expander(f"Рекомендации: {row['name']}"):

                recs = generate_risk_recommendations(row)

                for r in recs:
                    st.write("•", r)

        st.divider()

        # =========================
        # 📊 АНАЛИТИКА
        # =========================

        st.subheader("📊 Аналитика риска")

        for i in generate_risk_insights(df_risk):
            st.write("•", i)

        st.divider()

        # =========================
        # 🚨 ТОП РИСК
        # =========================

        st.subheader("🚨 Наиболее проблемные ученики")

        top_risk = df_sorted.head(5)

        st.dataframe(
            top_risk[["name", "risk_score", "risk_level"]],
            use_container_width=True
        )

        st.divider()

        # =========================
        # 🎯 ПРИОРИТЕТ ВМЕШАТЕЛЬСТВА
        # =========================

        st.subheader("🎯 Кому помогать в первую очередь")

        priority_df = get_risk_priority(df_risk)

        st.dataframe(
            priority_df.sort_values("priority_score", ascending=False)
            .head(5)[["name", "risk_score", "priority_score"]],
            use_container_width=True
        )

    elif menu == "Кластеры":

        st.title("👥 Кластеры")
        st.info("""
            Кластеры формируются автоматически на основе поведения учащихся:
            - успеваемость
            - посещаемость
            - мотивация
            - стресс

            Каждый кластер — это тип учеников с похожими характеристиками.
            """)
        
        st.subheader("  Таблица кластеров")
        st.dataframe(df[["name", "student_type", "cluster_explanation"]])
        st.divider()

        show_clusters(df)
        st.divider()

        show_cluster_heatmap(df)
        st.divider()

        st.subheader("🧠 Инсайты кластеров")
        for i in generate_cluster_insights(df):
            st.write("•", i)

        st.subheader("📌 Рекомендации классу")
        for r in generate_class_recommendations(df):
            st.write("•", r)

    # elif menu == "Инсайты":
    #     st.title("🧠 Общие инсайты")
    #     for i in generate_insights(df):
    #         st.write("•", i)

    elif menu == "Explainability":

        st.title("📊 Объяснение работы модели")

        st.markdown("""
    ### 🔍 Как работает система?

    Модель анализирует ключевые факторы:
    - посещаемость
    - выполнение домашней работы
    - средний балл
    - мотивация
    - уровень стресса

    ### 🧠 Принцип

    Модель сравнивает каждого ученика со средними значениями по выборке и выявляет:

    - факторы, ухудшающие результат
    - факторы, повышающие результат

    ### 📌 Пример

    Если:
    - низкая посещаемость  
    - низкая мотивация  

    → система снижает прогнозируемый балл

    ### 🎯 Зачем это нужно?

    Чтобы:
    - выявить слабые места ученика
    - дать рекомендации учителю
    - повысить качество образования
    """)

    elif menu == "Пользователи" and user["role"] == "admin":
        st.title("👥 Пользователи")
        
        st.title("👥 Управление пользователями")

        users = get_all_users()

        for u in users:
            col1, col2, col3 = st.columns(3)

            col1.write(u["login"])
            col2.write(u["role"])

            if col3.button(f"Удалить {u['login']}"):
                delete_user(u["login"])
                st.rerun()

        new_login = st.text_input("Логин")
        new_password = st.text_input("Пароль")
        role = st.selectbox("Роль", ["teacher", "admin"])
        class_name = st.text_input("Класс")

        if st.button("Создать"):
            create_user(new_login, new_password, role, class_name)
            st.success("Пользователь создан")

    elif menu == "Логи" and user["role"] == "admin":
        st.title("📜 Логи")

        logs = get_logs()
        st.dataframe(logs)
    
    elif menu == "Отчет":

        st.title("  Выгрузка отчета")

        # Фильтр по классу
        st.subheader("Фильтр по классу")

        if user["role"] == "admin":
            classes = df["class"].unique().tolist()
            selected_classes = st.multiselect("Выберите класс(ы) для отчёта", classes, default=classes)
            df_report = df[df["class"].isin(selected_classes)]
        else:
            # учитель видит только свой класс
            df_report = df[df["class"] == user["class"]]

        if st.button("Сгенерировать отчёт PDF"):

            from src.report_generator import generate_class_report
            from src.ai_strategy import generate_ai_strategy

            with st.spinner("Генерация отчета..." \
            "Подождите..."):

                result = generate_ai_strategy(df_report, pipeline, features)
                
            
                pdf_file = generate_class_report(df_report, result)
                    
                st.download_button(
                    label="Скачать PDF",
                    data=pdf_file,
                    file_name="AI_report.pdf",
                    mime="application/pdf"
                )


# ---------------- ROUTER ----------------
if not st.session_state.logged_in:
    login_page()
    st.stop()

main_app()