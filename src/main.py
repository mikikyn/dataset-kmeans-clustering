import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide", page_title="Instagram Segmentation")

base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, "Instagram visits clustering.csv")
model_path = os.path.join(base_path, "instagram_model.v1")
scaler_path = os.path.join(base_path, "scaler.v1")

@st.cache_data
def load_data():
    return pd.read_csv(csv_path)

try:
    df = load_data()
    kmeans = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except:
    st.error("Ошибка: Файлы модели, скалера или CSV не найдены в папке src!")
    st.stop()

st.title("Сегментация пользователей Instagram")
st.write("Метод: Кластеризация K-Means (Unsupervised Learning)")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.header("Карта сегментов")
    
    features = ['Instagram visit score', 'Spending_rank(0 to 100)']
    X_scaled = scaler.transform(df[features])
    df['cluster'] = kmeans.predict(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(df[features[0]], df[features[1]], c=df['cluster'], cmap='viridis', alpha=0.5)
    
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Центры групп')
    
    ax.set_xlabel("Счет посещений")
    ax.set_ylabel("Ранг трат")
    ax.legend()
    st.pyplot(fig)

with col_right:
    st.header("Анализ пользователя")
    st.write("Введите данные нового клиента:")
    
    in_visit = st.number_input("Счет посещений (0-110)", value=50)
    in_spending = st.number_input("Ранг трат (0-100)", value=50)
    
    if st.button("Определить сегмент"):
        new_data = pd.DataFrame([[in_visit, in_spending]], columns=features)
        new_scaled = scaler.transform(new_data)
        cluster_id = kmeans.predict(new_scaled)[0]
        
        st.subheader(f"Результат: Группа №{cluster_id}")
        
        if cluster_id == 0:
            st.info("Тип: 'Случайные прохожие' — редко заходят, мало тратят.")
        elif cluster_id == 1:
            st.success("Тип: 'Активные покупатели' — часто заходят и много тратят.")
        elif cluster_id == 2:
            st.warning("Тип: 'Зрители' — часто заходят, но почти ничего не покупают.")
        else:
            st.dark_code("Тип: 'Целевой сегмент' — средние показатели с потенциалом роста.")

st.divider()
st.header("Сырые данные")
st.dataframe(df.head(10), use_container_width=True)