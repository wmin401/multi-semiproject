# 필요한 라이브러리 import
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv("C:\\Users\\SAMSUNG\\Desktop\\DATA_2022년_국민여행조사_원자료.csv", encoding='cp949')

# 2022년 국내 여행자 데이터만 추출
df_2022 = df[df['D_TRA1_SYEAR'] == 2022]

# k-means 클러스터링
kmeans = KMeans(n_clusters=3, init='k-means++').fit(df_2022['D_TRA1_SMONTH'].values.reshape(-1,1))

# 클러스터링 결과를 새로운 열로 추가
df_2022['cluster'] = kmeans.labels_

# 클러스터별로 월별 데이터 집계
cluster_counts = df_2022.groupby('cluster')['D_TRA1_SMONTH'].value_counts().sort_index()

# 클러스터별로 시각화
for cluster in cluster_counts.index.levels[0]:
    month_counts = cluster_counts[cluster]
    plt.pie(month_counts, labels=month_counts.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85, wedgeprops=dict(width=0.3))
    
    # 도넛 모양 만들기
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.title(f'Travel Start Month Distribution - Cluster {cluster}')
    plt.show()
