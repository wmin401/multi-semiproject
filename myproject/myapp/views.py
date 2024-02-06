
# myapp/views.py
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from django.http import JsonResponse
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import base64
from io import BytesIO
from matplotlib.colors import ListedColormap
import threading  # 새로운 라이브러리 추가
import requests
from bs4 import BeautifulSoup
import pandas as pd
from IPython.display import display
import seaborn as sns
from sklearn import svm
import pickle
import joblib


def plot_thread(X_set, y_set, classifier, modelSet):
    buffer = BytesIO()
    cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_points = ListedColormap(['#FF0000', '#0000FF'])

    # Meshgrid 생성
    h = .02  # Step size in the mesh
    x_min, x_max = X_set[:, 0].min() - 1, X_set[:, 0].max() + 1
    y_min, y_max = X_set[:, 1].min() - 1, X_set[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 예측 결과 가져오기
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 결정 경계 및 데이터 포인트 시각화
    plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.3)
    scatter = plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set, cmap=cmap_points, edgecolor='k', s=20, label=('No Vehicle', 'Vehicle'))

    plt.title('Decision Boundary Plot ({})'.format(modelSet))
    plt.xlabel('Travel Frequency per Year')
    plt.ylabel('HouseHold Income')

    # Customize tick labels on the y-axis
    y_ticks = [-2, -1, 0, 1, 2]
    y_tick_labels = ['< ₩1,000,000', '< ₩2,000,000', '< ₩3,000,000', '< ₩4,000,000', '< ₩5,000,000']
    plt.yticks(y_ticks, y_tick_labels)

    # Customize tick labels on the x-axis
    x_ticks = [0, 2, 4, 6, 8, 10]
    x_tick_labels = [0, 1, 2, 3, 4, 5]
    plt.xticks(x_ticks, x_tick_labels)

    plt.legend(handles=scatter.legend_elements()[0], title="Vehicle Possession", labels=('No Vehicle', 'Vehicle'))
    plt.savefig(buffer, format='png')  # Change this line
    buffer.seek(0)
    plt.close()
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def naive_bayes(request):
    # 데이터 로딩
    file_path = 'DATA_2022년_국민여행조사_Bayes.csv'

    # 특정 컬럼 추출
    data = pd.read_csv(file_path, encoding='cp949', low_memory=False)

    data = data[['SA1_1', 'BINC1', 'DQ7']]

    # 칼럼 이름 변경
    culumn_names = {
                    'BINC1': '가구소득',
                    'SA1_1': '여행횟수',
                    'DQ7': '차량보유여부'
                    }

    data.rename(columns=culumn_names, inplace=True)

    # 인코딩
    label_encoder = LabelEncoder()
    data['여행횟수'] = label_encoder.fit_transform(data['여행횟수'])
    ordinal_encoder = OrdinalEncoder(categories=[['  100만원미만', '  100~200만원미만', '  200~300만원미만',
                                                  '  300~400만원미만', '  400~500만원미만', '  500~600만원미만', '  600만원 이상']])

    ordinal_encoder2 = OrdinalEncoder(categories=[['보유하지 않음', '보유하고 있음']])
    data['가구소득'] = ordinal_encoder.fit_transform(data[['가구소득']])
    data['차량보유여부'] = ordinal_encoder2.fit_transform(data[['차량보유여부']])

    # 결측값 제거
    x = data.iloc[:, [0, 1]].values
    y = data.iloc[:, 2].values

    # 트레이닝 데이터 셋, 테스트 데이터 셋 분류
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    # StandardScaler를 사용하여 특성 스케일링
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(x_test)

    # 트레이닝 데이터 세트 네이브 베이즈 분류기 모델 적용
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)



    # 테스트 데이터 세트를 활용하여 결과 예측
    y_pred = classifier.predict(X_test)

    # 정확도 측정
    accuracy = "{:.2f}".format(metrics.accuracy_score(y_test, y_pred))
    print(f'Accuracy: {accuracy}')
    print("\n분류 보고서:\n", classification_report(y_test, y_pred))

    # 메트릭 측정
    cm = confusion_matrix(y_test, y_pred)

    plot_image = plot_thread(X_train, y_train, classifier, "Training Set")
    plot_image2 = plot_thread(X_test, y_test, classifier, "Testing Set")

    # Return the base64-encoded image, accuracy, and classification report in JSON response
    response_data = {
        'plot_image': plot_image,
        'plot_image2': plot_image2,
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred)
    }


    # Return the base64-encoded image in JSON response
    return JsonResponse(response_data)

def k_means_clustering(request):
    # LabelEncoder 객체 생성
    encoder = LabelEncoder()

    # 데이터 불러오기
    df = pd.read_csv("DATA_2022년_국민여행조사_원자료_1.csv", encoding='cp949', low_memory=False)

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
    return JsonResponse({"encoded_image": encoded_image})


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

def svm_function(request):
    # 모델 학습
    model = svm.SVC()
    filename = '전처리완료그래프.csv'
    df = pd.read_csv(filename, encoding='cp949')
    df = df.dropna()

    def convert_number_of_people(value):
        if value in [3, 4, 5, 6, 7, 8, 9]:
            return 3
        elif value >= 10:
            return 4
        else:
            return value

    df['Number_of_people'] = df['Number_of_people'].apply(convert_number_of_people)
    df = df.drop(['SPOT', 'SEX', 'AGE', 'Total accommodation fee'], axis=1)
    Y_data = df['Accommodation']
    X_data = df.drop('Accommodation', axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

    # 모델 학습 및 저장
    model.fit(X_train, Y_train)
    joblib.dump(model, 'svm_model.pkl')

    # 저장된 모델 로드
    loaded_model = joblib.load('svm_model.pkl')
    predictions = loaded_model.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    #mean_squared_error_result = mean_squared_error(Y_test, predictions)
    classification_result = classification_report(Y_test, predictions)


    with open('xx.pkl', 'rb') as f:
        xx = pickle.load(f)
    with open('yy.pkl', 'rb') as f:
        yy = pickle.load(f)
    with open('Z.pkl', 'rb') as f:
        Z = pickle.load(f)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=Y_train, s=20, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.xlabel('Total transportation cost')
    plt.ylabel('Number_of_people')
    plt.title('SVM Decision Boundary with Best Parameters')
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    image = BytesIO()
    plt.savefig(image, format='png')
    image.seek(0)
    plot_image = base64.b64encode(image.getvalue()).decode()
    plt.close()

    response_data = {
        'plot_image1': plot_image,
        'accuracy': accuracy,

        'classification_report': classification_result
    }

    return JsonResponse(response_data)

