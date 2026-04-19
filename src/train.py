import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "Instagram visits clustering.csv")

df = pd.read_csv(file_path)

features = ['Instagram visit score', 'Spending_rank(0 to 100)']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

joblib.dump(kmeans, os.path.join(base_path, 'instagram_model.v1'))
joblib.dump(scaler, os.path.join(base_path, 'scaler.v1'))
print("Модель кластеризации и скалер сохранены!")

plt.figure(figsize=(10, 6))
plt.scatter(df[features[0]], df[features[1]], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.scatter(
    scaler.inverse_transform(kmeans.cluster_centers_)[:, 0], 
    scaler.inverse_transform(kmeans.cluster_centers_)[:, 1], 
    s=200, c='red', marker='X', label='Центры групп'
)

plt.title("Кластеризация пользователей Instagram")
plt.xlabel("Счет посещений (Visit Score)")
plt.ylabel("Ранг трат (Spending Rank)")
plt.legend()
plt.grid(True)
plt.show()

print("\nКоличество людей в каждой группе:")
print(df['cluster'].value_counts()) 