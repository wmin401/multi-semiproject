# 필요한 라이브러리 import
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 데이터 불러오기
# low_memory=False 옵션 추가: 데이터 타입 추론을 하지 않고 메모리를 효율적으로 사용
# nrows=10000 옵션 추가: 데이터의 크기를 줄이기 위해 처음 10000개 행만 불러오기
df = pd.read_csv("C:\\Users\\SAMSUNG\\Desktop\\DATA_2022년_국민여행조사_원자료.csv", encoding='cp949', low_memory=False, nrows=10000)

# 필요한 칼럼만 추출
df = df[['D_TRA1_CASE', 'D_TRA1_COST']]

# 결측치 제거
df = df.dropna()

# 여행 유형 칼럼을 숫자로 인코딩
le = LabelEncoder()
df['D_TRA1_CASE'] = le.fit_transform(df['D_TRA1_CASE'].astype(str))

# 여행 비용 칼럼을 5만원 단위로 반올림
df['D_TRA1_COST'] = np.round(df['D_TRA1_COST'].astype(float) / 50000) * 50000

# 데이터 정규화
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# k-means 클러스터링
kmeans = KMeans(n_clusters=3, init='k-means++').fit(df_scaled)

# 클러스터링 결과 추가
df['Cluster'] = kmeans.labels_

# scatter plot 그리기
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(df['D_TRA1_CASE'], df['D_TRA1_COST'], c=df['Cluster'], cmap='viridis')
ax.set_xlabel('Travel Type', fontsize=15)
ax.set_ylabel('Travel Cost', fontsize=15)
ax.set_title('K-means Clustering', fontsize=20)
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
plt.show()  # 그래프를 화면에 표시
