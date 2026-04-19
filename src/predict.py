import joblib
import pandas as pd
import os

base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'instagram_model.v1')
scaler_path = os.path.join(base_path, 'scaler.v1')

try:    
    kmeans = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except:
    print("Ошибка: Сначала запустите train.py, чтобы создать файлы модели и скалера.")
    exit()

print("СИСТЕМА КЛАСТЕРИЗАЦИИ ПОЛЬЗОВАТЕЛЕЙ INSTAGRAM")

try:
    visit_score = float(input("Введите счет посещений (Visit Score, например 0-100): "))
    spending_rank = float(input("Введите ранг трат (Spending Rank, например 0-100): "))

    input_data = pd.DataFrame(
        [[visit_score, spending_rank]],
        columns=['Instagram visit score', 'Spending_rank(0 to 100)']
    )
    input_scaled = scaler.transform(input_data)

    cluster = kmeans.predict(input_scaled)[0]

    print(f"\nРЕЗУЛЬТАТ АНАЛИЗА")
    print(f"Пользователь отнесен к группе №: {cluster}")

    if cluster == 0:
        print("Описание: Малоактивный пользователь с низкими тратами.")
    elif cluster == 1:
        print("Описание: Активный пользователь, высокий потенциал для покупок.")

except Exception as e:
    print(f"Ошибка ввода: {e}")