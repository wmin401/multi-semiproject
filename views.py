# myapp/views.py

from django.http import JsonResponse

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render
from io import BytesIO
import base64
import matplotlib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

import requests
from bs4 import BeautifulSoup
import pandas as pd
from IPython.display import display

matplotlib.use('Agg')  # Matplotlib이 메인 스레드 이외에서도 사용될 수 있도록 설정

def k_means_clustering(request):
        
    # LabelEncoder 객체 생성
    encoder = LabelEncoder()

    # 데이터 불러오기
    df = pd.read_csv("C:/Users/SAMSUNG/Desktop/DATA_2022년_국민여행조사_원자료_1.csv", encoding='cp949', low_memory=False)

    # 필요한 칼럼 추가로 추출
    df = df[['D_TRA1_CASE', 'D_TRA1_COST', 'D_TRA1_ONE_COST', 'A9C', 'A9D']]

    # 결측치 제거
    df = df.dropna()

    # 'A9C'와 'A9D' 칼럼이 문자열이라면 숫자로 변환
    if df['A9C'].dtype == 'object':
        df['A9C'] = encoder.fit_transform(df['A9C'])
    if df['A9D'].dtype == 'object':
        df['A9D'] = encoder.fit_transform(df['A9D'])

    # 여행 비용 칼럼과 1인당 여행비용 칼럼을 10만원 단위로 반올림
    df['D_TRA1_COST'] = np.round(df['D_TRA1_COST'].astype(float) / 100000) * 100000
    df['D_TRA1_ONE_COST'] = np.round(df['D_TRA1_ONE_COST'].astype(float) / 100000) * 100000

    # 필요한 칼럼만으로 새로운 데이터프레임 생성
    df_new = df[['D_TRA1_ONE_COST', 'D_TRA1_COST', 'A9C', 'A9D']]

    # 데이터 정규화
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_new), columns=df_new.columns)

    # k-means 클러스터링
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42).fit(df_scaled)

    # 클러스터링 결과 추가
    df['Cluster'] = kmeans.labels_

        # subplot 생성
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))

    # scatter plot 그리기 for 'A9C'
    scatter = axs[0, 0].scatter(df['D_TRA1_ONE_COST'], df['A9C'], c=df['Cluster'], cmap='viridis')
    axs[0, 0].set_xlabel('Travel Cost per Person', fontsize=15)
    axs[0, 0].set_ylabel('Accommodation Cost', fontsize=15)
    axs[0, 0].set_title('K-means Clustering for Accommodation Cost', fontsize=20)
    legend1 = axs[0, 0].legend(*scatter.legend_elements(), title="Clusters")

    # scatter plot 그리기 for 'A9D'
    scatter = axs[0, 1].scatter(df['D_TRA1_ONE_COST'], df['A9D'], c=df['Cluster'], cmap='viridis')
    axs[0, 1].set_xlabel('Travel Cost per Person', fontsize=15)
    axs[0, 1].set_ylabel('Restaurant Cost', fontsize=15)
    axs[0, 1].set_title('K-means Clustering for Restaurant Cost', fontsize=20)
    legend1 = axs[0, 1].legend(*scatter.legend_elements(), title="Clusters")

    # '여행 유형' 칼럼을 숫자로 변환
    df['D_TRA1_CASE'] = encoder.fit_transform(df['D_TRA1_CASE'])

    # 상관계수 행렬 계산
    corr_matrix = df.corr()

    # Heatmap으로 상관계수 행렬을 시각화
    sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', ax=axs[1, 0])

    # Elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(df_scaled)
        wcss.append(kmeans.inertia_)

    axs[1, 1].plot(range(1, 11), wcss, marker='o', linestyle='--')
    axs[1, 1].set_xlabel('Number of Clusters', fontsize=15)
    axs[1, 1].set_ylabel('WCSS', fontsize=15)
    axs[1, 1].grid(True)

    plt.tight_layout()

     # 그래프를 이미지로 변환
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    # 이미지를 Base64로 인코딩
    encoded_image = base64.b64encode(image_stream.read()).decode("utf-8")
    image_stream.close()

    # 시각화된 이미지를 템플릿에 전달
    return JsonResponse( {"encoded_image":encoded_image})

def web_crawling(request):
    query = '국내 여행'
    url = f'https://search.naver.com/search.naver?where=view&sm=tab_jum&query={query}'

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    print(response.status_code)
    results = soup.select('a.title_link')

    data = []
    for result in results:
        title = result.text
        link = result['href']
        data.append([title, link])

    df = pd.DataFrame(data, columns=['Title', 'Link'])
    df.to_csv('holiday_recommendations.csv', index=False, encoding='utf-8-sig')
    df = pd.read_csv('holiday_recommendations.csv')
    display(df)
    
    return 0