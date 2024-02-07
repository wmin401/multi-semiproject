# 필요한 라이브러리 import
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 데이터 불러오기
df = pd.read_csv("C:/Users/SAMSUNG/Desktop/DATA_2022년_국민여행조사_원자료.csv", encoding='cp949', low_memory=False)

# 필요한 칼럼 추가
df = df[['D_TRA1_CASE', 'D_TRA1_COST', 'D_TRA1_ONE_COST', 'D_TRA1_1_Q6']]

# 결측치 제거
df = df.dropna()

# 숙소 유형을 카테고리형 데이터로 변환
df['D_TRA1_1_Q6'] = df['D_TRA1_1_Q6'].astype('category')

# 숙소 유형별로 그룹화
grouped = df.groupby('D_TRA1_1_Q6', observed=True)

# 그래프 생성을 위한 색상 리스트
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

# 숙소 유형별로 클러스터링 진행
fig, ax = plt.subplots(figsize=(8, 8))
for i, (name, group) in enumerate(grouped):
    # 데이터 정규화
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(group[['D_TRA1_ONE_COST', 'D_TRA1_COST']]), columns=['D_TRA1_ONE_COST', 'D_TRA1_COST'])

    # k-means 클러스터링
    kmeans = KMeans(n_clusters=3, init='k-means++').fit(df_scaled)

    # 클러스터링 결과 추가
    group['Cluster'] = kmeans.labels_

    # scatter plot 그리기
    scatter = ax.scatter(group['D_TRA1_ONE_COST'], group['D_TRA1_COST'], c=colors[i%len(colors)], label=name, alpha=0.5, edgecolors=None)

ax.set_xlabel('Travel Cost per Person', fontsize=15)
ax.set_ylabel('Total Travel Cost', fontsize=15)
ax.set_title('K-means Clustering by Accommodation Type', fontsize=20)
ax.legend()
plt.show()
