# 필요한 라이브러리를 불러옵니다.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# csv 파일을 읽어옵니다.
df = pd.read_csv("C:/Users/SAMSUNG/Desktop/20240126145505_전국_202301-202312_데이터랩_다운로드/20240126145505_방문자 체류특성.csv", encoding='cp949')

# KMeans 클러스터링을 위한 데이터를 준비합니다.
# '평균 체류시간', '평균 숙박일수' 두 가지 특징을 사용하여 클러스터링을 진행합니다.
X = df[['평균 체류시간', '평균 숙박일수']]

# KMeans 모델을 학습시킵니다. 클러스터의 수는 도의 수인 13으로 설정합니다.
kmeans = KMeans(n_clusters=13, random_state=0).fit(X)

# 클러스터링 결과를 'cluster' 열로 추가합니다.
df['cluster'] = kmeans.labels_

# 클러스터링 결과를 시각화합니다.
plt.figure(figsize=(10, 7))
sns.scatterplot(x='평균 체류시간', y='평균 숙박일수', hue='cluster', data=df, palette='Set2')
plt.title('KMeans Clustering')
plt.show()
