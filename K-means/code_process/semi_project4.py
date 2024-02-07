# 필요한 라이브러리 import
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 데이터 불러오기
df = pd.read_csv("C:\\Users\\SAMSUNG\\Desktop\\DATA_2022년_국민여행조사_원자료.csv", encoding='cp949')

# 여행 유형 칼럼을 문자열로 변경
df['D_TRA1_CASE'] = df['D_TRA1_CASE'].map({1: '국내 관광/휴양 여행', 2: '국내 가족/친지/친구 방문 여행 - 관광/휴양 활동 포함', 3: '국내 단순 가족/친지/친구 방문 - 관광/휴양 활동이 포함되지 않음'})

# 필요한 칼럼만 추출
df = df[['D_TRA1_CASE', 'D_TRA1_SMONTH', 'D_TRA1_COST']]

# 결측치 제거
df = df.dropna(subset=['D_TRA1_SMONTH', 'D_TRA1_COST'])

# 여행 비용 칼럼을 5만원 단위로 반올림
df['D_TRA1_COST'] = np.round(df['D_TRA1_COST'] / 50000) * 50000

# 여행 유형 칼럼을 숫자로 인코딩
le = LabelEncoder()
df['D_TRA1_CASE'] = le.fit_transform(df['D_TRA1_CASE'])

# 데이터 정규화
scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# k-means 클러스터링
kmeans = KMeans(n_clusters=3, init='k-means++').fit(df)

# 클러스터링 결과 추가
df['Cluster'] = kmeans.labels_

# PCA로 차원 축소
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df)

# 클러스터링 결과 추가
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
principalDf['Cluster'] = df['Cluster']

# scatter plot 그리기
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 Component PCA', fontsize=20)

colors = ['r', 'g', 'b']
clusters = [0, 1, 2]
for cluster, color in zip(clusters, colors):
    indicesToKeep = principalDf['Cluster'] == cluster
    ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1'], principalDf.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
ax.legend(clusters)
ax.grid()

plt.show()  # 그래프를 화면에 표시
