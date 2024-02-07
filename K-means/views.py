
# myapp/views.py
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from django.http import JsonResponse
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics, svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import base64
from io import BytesIO
from matplotlib.colors import ListedColormap
import threading  # 새로운 라이브러리 추가
import requests
from bs4 import BeautifulSoup
import pandas as pd
from IPython.display import display
import seaborn as sns
import pickle
import joblib
import urllib
from itertools import cycle
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.image as mpimg
import matplotlib.font_manager as fm
from PIL import Image
import io



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
    print("K-mean-clustering Success!")
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

    print("k-means-clustering Success!")

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
    print("svm Sucess!")
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

def plot_heatmap(correlation_matrix):
    plt.figure(figsize=(10,10))
    sns.heatmap(correlation_matrix, annot=True)

    # 그래프를 이미지 파일로 저장
    image = BytesIO()
    plt.savefig(image, format='png')
    image.seek(0)
    # 이미지 파일을 base64로 인코딩
    plot_image = base64.b64encode(image.getvalue()).decode()

    plt.close()

    return plot_image

def plot_mse(X_test, y_test, y_pred, mse):
    plt.figure(figsize=(15, 7))
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred, color='red', label='Predicted')
    plt.title('Linear Regression using MSE\nMSE: {0:.2f}'.format(mse))
    plt.xlabel('Cost')
    plt.ylabel('the number of days of travel')
    plt.yticks(ticks=np.arange(1, max(y_test) + 1, step=1))  # y축 눈금 설정
    plt.legend()

    # 그래프를 이미지 파일로 저장
    image = BytesIO()
    plt.savefig(image, format='png')
    image.seek(0)
    # 이미지 파일을 base64로 인코딩
    plot_image = base64.b64encode(image.getvalue()).decode()

    plt.close()

    return plot_image

def plot_training_set(X_train, y_train, regressor):
    plt.figure(figsize=(15, 7))
    plt.scatter(X_train, y_train, color='blue')
    plt.plot(X_train, regressor.predict(X_train), color='red')
    plt.title('Training set')
    plt.xlabel('average monthly income')
    plt.ylabel('Expenditure amount')

    # 그래프를 이미지 파일로 저장
    image = BytesIO()
    plt.savefig(image, format='png')
    image.seek(0)
    # 이미지 파일을 base64로 인코딩
    plot_image = base64.b64encode(image.getvalue()).decode()

    plt.close()

    return plot_image

def plot_test_set(X_test, y_test, regressor):
    plt.figure(figsize=(15, 7))
    plt.scatter(X_test, y_test, color='blue')
    plt.plot(X_test, regressor.predict(X_test), color='red') # 변경: 테스트 데이터에 대한 예측값 사용
    plt.title('Test set')
    plt.xlabel('average monthly income')
    plt.ylabel('Expenditure amount')

    # 그래프를 이미지 파일로 저장
    image = BytesIO()
    plt.savefig(image, format='png')
    image.seek(0)
    # 이미지 파일을 base64로 인코딩
    plot_image = base64.b64encode(image.getvalue()).decode()

    plt.close()

    return plot_image

def Linear_Regression(request):
    print("Linear Regression Success!")
    print("test")
    # 데이터 로딩
    file_path = 'LinearRegression.csv'

    # 특정 컬럼 추출
    dataset = pd.read_csv(file_path, encoding='cp949', low_memory=False)

    # D_TRA1_S_Day 컬럼의 NaN값 삭제
    dataset = dataset.dropna(subset=['D_TRA1_S_Day'])

    # 업데이트된 데이터셋 사용
    selected_columns = dataset[['D_TRA1_COST', 'D_TRA1_S_Day', 'D_TRA1_ONE_COST']]

    # 데이터 컬럼명 변경
    dataset = dataset.rename(columns={'D_TRA1_COST': '지출액', 'D_TRA1_S_Day': '여행일', 'D_TRA1_ONE_COST': '1인지출비용'})

    # 선택된 컬럼들의 상관관계 계산
    correlation_matrix = selected_columns.corr()

    # '여행일'을 독립 변수로, '1인지출비용'을 종속 변수로 설정
    X = dataset['1인지출비용'].values.reshape(-1, 1)
    y = dataset['여행일'].values.reshape(-1, 1)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 데이터 표준화
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 선형 회귀 모델 생성 및 학습
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # 예측
    y_pred = regressor.predict(X_test)

    # MSE 계산
    mse = metrics.mean_squared_error(y_test, y_pred)

    plot_image1 = plot_heatmap(correlation_matrix)
    plot_image2 = plot_training_set(X_train, y_train, regressor)
    plot_image3 = plot_test_set(X_test, y_test, regressor)
    plot_image4 = plot_mse(X_test, y_test, y_pred, mse)

    # 정확도 측정
    accuracy = "{:.2f}".format(metrics.mean_squared_error(y_test, y_pred))

    # 분류 보고서 생성
    metrics.mean_squared_error_result = metrics.mean_squared_error(y_test, y_pred)

    response_data = {
        'plot_image1': plot_image1,
        'plot_image2': plot_image2,
        'plot_image3': plot_image3,
        'plot_image4': plot_image4,
        'accuracy': accuracy,
        'classification_report': metrics.mean_squared_error_result
    }

    return JsonResponse(response_data)

def Logistic_Regression(request):
    # 이미지 파일을 열기
    img = Image.open('LogisticRegression.png')

    # BytesIO 객체 생성
    byte_arr = io.BytesIO()

    # 이미지를 BytesIO 객체에 저장
    img.save(byte_arr, format='PNG')

    # BytesIO의 내용을 base64로 인코딩
    plot_image1 = base64.b64encode(byte_arr.getvalue()).decode()

    response_data = {
        'plot_image1': plot_image1
    }

    return JsonResponse(response_data)


    # # 데이터 로딩
    # file_path = 'LogisticRegression.csv'
    #
    # # 특정 컬럼 추출
    # dataset = pd.read_csv(file_path, encoding='cp949', low_memory=False)
    #
    # # 문자열 데이터 변환: 원-핫 인코딩
    # dataset_encoded = pd.get_dummies(dataset,
    #                                  columns=['A5_1', 'A5_2', 'A5_3', 'A5_4', 'A5_5', 'A5_6', 'A5_7', 'A5_8', 'A5_9',
    #                                           'A5_10', 'A5_11', 'A5_12', 'A5_13', 'A5_14', 'A5_15', 'A5_16', 'A5_17',
    #                                           'A5_18', 'A5_19', 'A5_20', 'A5_21'])
    #
    # # NaN 값 처리: 대체값 할당
    # dataset_encoded = dataset_encoded.fillna(0)  # NaN 값을 0으로 대체
    #
    # # 출력 행 수를 230개로 설정
    # pd.set_option('display.max_rows', 230)
    #
    # # 레이블 인코더를 생성합니다.
    # label_encoder = LabelEncoder()
    #
    # # D_TRA1_1_SPOT 컬럼을 레이블 인코딩합니다.
    # dataset['D_TRA1_1_SPOT_ENCODED'] = label_encoder.fit_transform(dataset['D_TRA1_1_SPOT'])
    #
    # # 인코딩 결과를 확인합니다.
    # encoded_values = dataset[['D_TRA1_1_SPOT', 'D_TRA1_1_SPOT_ENCODED']].drop_duplicates().sort_values(
    #     'D_TRA1_1_SPOT_ENCODED')
    #
    # # A5_1부터 A5_21까지의 열을 원-핫 인코딩하여 새로운 열로 추가합니다.
    # encoded_columns = pd.get_dummies(dataset[['A5_1', 'A5_2', 'A5_3', 'A5_4', 'A5_5', 'A5_6', 'A5_7', 'A5_8', 'A5_9',
    #                                           'A5_10', 'A5_11', 'A5_12', 'A5_13', 'A5_14', 'A5_15', 'A5_16', 'A5_17',
    #                                           'A5_18', 'A5_19', 'A5_20', 'A5_21']])
    #
    # # 새로운 데이터프레임 생성
    # new_dataset = pd.concat([encoded_columns, dataset['D_TRA1_1_SPOT_ENCODED']], axis=1)
    #
    # # 컬럼명 변경을 위한 딕셔너리 생성
    # column_mapping = {
    #     'A5_1_자연 및 풍경감상': '1',
    #     'A5_2_음식관광(지역 맛집 등)': '2',
    #     'A5_3_야외 위락 및 스포츠, 레포츠 활동': '3',
    #     'A5_4_역사 유적지 방문': '4',
    #     'A5_5_테마파크, 놀이시설, 동/식물원 방문': '5',
    #     'A5_6_휴식/휴양': '6',
    #     'A5_7_온천/스파': '7',
    #     'A5_7_휴식/휴양': '7',
    #     'A5_8_쇼핑': '8',
    #     'A5_9_지역 문화예술/공연/전시시설 관람': '9',
    #     'A5_10_스포츠 경기관람': '10',
    #     'A5_11_지역 축제/이벤트 참가': '11',
    #     'A5_12_교육/체험 프로그램 참가': '12',
    #     'A5_13_종교/성지순례': '13',
    #     'A5_14_카지노, 경마, 경륜 등': '14',
    #     'A5_15_시티투어': '15',
    #     'A5_16_드라마 촬영지 방문': '16',
    #     'A5_17_유흥/오락': '17',
    #     'A5_18_가족/친지/친구 방문': '18',
    #     'A5_19_회의참가/시찰': '19',
    #     'A5_20_교육/훈련/연수': '20',
    #     'A5_21_기타': '21',
    #     'D_TRA1_1_SPOT_ENCODED': 'travel destination'
    # }
    #
    # # 컬럼명 변경
    # new_dataset = new_dataset.rename(columns=column_mapping)
    #
    # # 입력 변수와 출력 변수 설정
    # X = new_dataset.iloc[:, :-1]  # '1'부터 '21'까지의 열을 입력 변수로 설정
    # y = new_dataset['travel destination']  # 'travel destination' 열을 출력 변수로 설정
    #
    # # 최적의 파라미터로 로지스틱 회귀 모델 생성
    # model = LogisticRegression(C=1, penalty='l2')
    #
    # # 모델 훈련
    # model.fit(X, y)
    #
    # # 예측
    # y_pred = model.predict(X)
    #
    # # 결과 평가
    # accuracy = accuracy_score(y, y_pred)
    #
    # # 예측 결과를 DataFrame으로 변환
    # y_pred_df = pd.DataFrame(y_pred, columns=['Predicted Destination'])
    #
    # # 원래의 데이터셋에 예측 결과 추가
    # new_dataset_with_prediction = pd.concat([new_dataset, y_pred_df], axis=1)
    #
    # # 'travel destination'의 빈도 계산
    # destination_counts = new_dataset['travel destination'].value_counts()
    #
    # # 가장 인기 있는 여행지 3개 뽑기
    # top_destinations = destination_counts.nlargest(4).index.tolist()
    #
    # # 229번 여행지 제외
    # if 229 in top_destinations:
    #     top_destinations.remove(229)
    #
    # # 각 여행지에서 가장 인기 있는 여행 유형 3개 뽑기
    # top_3_travel_types_per_top_destination = {}
    # for destination in top_destinations:
    #     # 특정 여행지에 해당하는 행만 선택
    #     destination_data = new_dataset[new_dataset['travel destination'] == destination]
    #
    #     # 여행 유형별 빈도 계산
    #     travel_type_counts = destination_data.iloc[:, :-2].sum()
    #
    #     # 가장 빈도가 높은 여행 유형 3개 뽑기
    #     top_3_travel_types = travel_type_counts.nlargest(3).index.tolist()
    #
    #     top_3_travel_types_per_top_destination[destination] = top_3_travel_types
    #
    # print(top_3_travel_types_per_top_destination)
    #
    # # 여행 유형과 빈도 데이터를 담을 리스트
    # travel_types = []
    # frequencies = []
    #
    # # 각 여행지에서 가장 인기 있는 여행 유형 3개의 빈도 계산
    # for destination in top_destinations:
    #     destination_data = new_dataset[new_dataset['travel destination'] == destination]
    #     travel_type_counts = destination_data.iloc[:, :-2].sum()
    #     top_3_travel_types = travel_type_counts.nlargest(3)
    #
    #     # 여행 유형과 빈도 데이터 추가
    #     travel_types.extend(top_3_travel_types.index)
    #     frequencies.extend(top_3_travel_types.values)
    #
    # # 막대 그래프 그리기
    # x = np.arange(len(top_destinations))  # 여행지의 위치
    # width = 0.2  # 막대의 너비
    #
    # fig, ax = plt.subplots()
    # for i in range(3):
    #     if travel_types[i] == '1':
    #         label = 'Appreciation of Nature'
    #     elif travel_types[i] == '6':
    #         label = 'Rest/Recreation'
    #     elif travel_types[i] == '2':
    #         label = 'Visiting Good Restaurants'
    #     else:
    #         label = 'Travel Type ' + travel_types[i]
    #
    #     ax.bar(x - width + i * width, frequencies[i::3], width, label=label)
    #
    # # 그래프 설정
    # ax.set_xlabel('Destination')
    # ax.set_ylabel('Frequency')
    # ax.set_title('Top 3 Travel Types for Each Top 3 Destinations')
    # ax.set_xticks(x)
    # ax.set_xticklabels(['Gangneung', 'Gyeongju', 'Jeju'])
    # ax.legend()
    #
    # # 그래프를 이미지 파일로 저장
    # image = BytesIO()
    # plt.savefig(image, format='png')
    # image.seek(0)
    #
    # # 이미지 파일을 base64로 인코딩
    # plot_image1 = base64.b64encode(image.getvalue()).decode()
    #
    # # 그래프 닫기
    # plt.close()
    #
    # response_data = {
    #     'plot_image1': plot_image1,
    #     'accuracy': accuracy
    # }
    #
    # return JsonResponse(response_data)


def RandomForest(request):
    # 데이터 로딩
    # file_path = 'rf_data.csv'
    # train = pd.read_csv(file_path, low_memory=False)

    # #인코딩, 변수지정
    # X = pd.get_dummies(train.drop('여행유형', axis=1))
    # y = train['여행유형']

    # # 트레이닝 데이터 셋, 테스트 데이터 셋 분류
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # #하이퍼파라미터 설정 및 랜덤포레스트 훈련
    # param = {'n_estimators': 490, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 10, 'max_depth': None, 'criterion': 'gini', 'bootstrap': True}

    # classifier = RandomForestClassifier(**param, random_state=42)
    # classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)

    # # 정확도측정
    # accuracy = "{:.2f}".format(metrics.accuracy_score(y_test, y_pred))
    # print(f'Accuracy: {accuracy}')
    # print("\n분류 보고서:\n", classification_report(y_test, y_pred))

    # importances = classifier.feature_importances_
    # indices = np.argsort(importances)[::-1][:10]

    # #한글폰트
    # font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    # font = fm.FontProperties(fname=font_path, size=10)
    # plt.rc('font', family='NanumGothic')
    # plt.rcParams["font.family"] = 'NanumGothic'
    # plt.rcParams['axes.unicode_minus'] = False

    # # Roc곡선 그래프를 위한 인코딩
    # le = preprocessing.LabelEncoder()
    # y_encoded = le.fit_transform(y)

    # y_bin = label_binarize(y_test, classes=np.unique(y_test))
    # n_classes = y_bin.shape[1]

    # # 예측 확률
    # y_score = classifier.predict_proba(X_test)

    fig, axs = plt.subplots(figsize=(13, 10))  # 충분한 subplot 생성

    # # # 1. 트리 그래프
    # # plot_tree(
    # #     rf.estimators_[0],
    # #     feature_names=X.columns,
    # #     class_names=y.unique(),
    # #     filled=True,
    # #     ax=axs[1, 0]
    # # )

    # 로딩 시간 관계로 이미지로 대체
    # img = mpimg.imread('final_tree.png')
    # axs[0, 0].imshow(img)
    # axs[0, 0].axis('off')

    img = mpimg.imread('fianl_01.png')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # # 2. 피쳐 중요도
    # axs[0, 1].set_title('피쳐 중요도')
    # axs[0, 1].bar(range(10), importances[indices], align='center')
    # axs[0, 1].set_xticks(range(10))
    # axs[0, 1].set_xticklabels(X.columns[indices], rotation=90)

    # # 3. Confusion Matrix
    # cm = confusion_matrix(y_test, y_pred)
    # sns.heatmap(cm, annot=True, fmt=".0f", ax=axs[1, 0])
    # axs[1, 0].set_title('Confusion Matrix')
    # axs[1, 0].set_xlabel('Predicted')
    # axs[1, 0].set_ylabel('Actual')

    # # 4. ROC
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'skyblue', 'pink'])
    # for i, color in zip(range(n_classes), colors):
    #     fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
    #     roc_auc = auc(fpr, tpr)
    #     original_label = le.inverse_transform([i])
    #     axs[1, 1].plot(fpr, tpr, color=color, lw=2,
    #                 label='ROC {0} (area = {1:0.2f})'
    #                 ''.format(original_label, roc_auc))

    # axs[1, 1].plot([0, 1], [0, 1], 'k--', lw=2)
    # axs[1, 1].set_xlim([0.0, 1.0])
    # axs[1, 1].set_ylim([0.0, 1.05])
    # axs[1, 1].set_xlabel('FPR')
    # axs[1, 1].set_ylabel('TPR')
    # axs[1, 1].set_title('ROC')
    # axs[1, 1].legend(loc="lower right")

    plt.tight_layout()

    # 그래프를 이미지로 변환
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    # 이미지를 Base64로 인코딩
    encoded_image = base64.b64encode(image_stream.read()).decode("utf-8")
    image_stream.close()

    rf_response_data = {
        'encoded_image': encoded_image,

    }

    return JsonResponse(rf_response_data)

def DecisionTree(request):
    print("Decision Tree Success!")
    # CSV 파일 경로
    file_path = 'COST_and_ONECOST.csv'

    # CSV 파일 읽고, 인코딩 설정
    df = pd.read_csv(file_path, encoding='cp949', low_memory=False)

    # 필요한 컬럼 추출 (COST = 여행 총 경비 / ONE_COST = 1인 지출 비용)
    df1 = df[['D_TRA1_COST', 'D_TRA1_ONE_COST']].copy()
    # 결측치 처리 둘중에 선택 전자는 평균값으로 / 후자는 결측치행삭제 / 전처리작업
    #df1.fillna(df1.mean(), inplace=True)
    df1.dropna(inplace=True)


    # 데이터 스케일링
    scaler = StandardScaler()
    df1_scaled = pd.DataFrame(scaler.fit_transform(df1), columns=df1.columns)
    # 데이터프레임 내용 확인

    # 각 변수 분포 확인
    plt.figure(figsize=(6, 4))
    plt.hist(df1_scaled, bins=50)
    plt.title("Histogram of Variable Distribution")
    plt.xlabel("COST")
    plt.ylabel("People")
    plt.xticks([])
    plt.yticks([])

    image = BytesIO()
    plt.savefig(image, format='png')
    image.seek(0)
    plot_image1 = base64.b64encode(image.getvalue()).decode()
    plt.close()

    # 토탈 상관계수
    plt.figure(figsize=(6, 4))
    sl_col = df1_scaled[['D_TRA1_COST', 'D_TRA1_ONE_COST']]
    sns.heatmap(sl_col.corr(), annot=True, fmt=".2f")

    image = BytesIO()
    plt.savefig(image, format='png')
    image.seek(0)
    plot_image2 = base64.b64encode(image.getvalue()).decode()
    plt.close()

    # X = 독립변수 / y = 종속변수 설정
    X = df1_scaled[['D_TRA1_COST']]
    y = df1_scaled[['D_TRA1_ONE_COST']]
    # 학습데이터, 테스트데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # RandomizedSearchCV 파라미터 설정
    param_distributions = {
        'max_depth': range(1, 20),
        'min_samples_split': range(2, 20),
        'min_samples_leaf': range(1, 20)
    }

    # RandomizedSearchCV를 사용한 모델 생성 및 학습
    rand_search = RandomizedSearchCV(DecisionTreeRegressor(random_state=1), param_distributions, n_iter=100, cv=3, scoring='neg_mean_squared_error', random_state=1)
    rand_search.fit(X_train, y_train)
    # 최적의 파라미터와 점수 출력

    # 최적 하이퍼 파라미터로 학습된 모델 가져오기
    DTR = rand_search.best_estimator_

    # 테스트 데이터로 예측
    y_pred = DTR.predict(X_test)


    # 모델 성능 평가
    mse = mean_squared_error(y_test, y_pred)

    # MAE 계산
    mae = mean_absolute_error(y_test, y_pred)

    # R-Squared 계산
    r2 = r2_score(y_test, y_pred)

    # 시각화
    plt.figure(figsize=(10, 7))
    tree.plot_tree(DTR, max_depth=2, feature_names=X.columns, filled=True)
    plt.title("Decision Tree Visualization")

    image = BytesIO()
    plt.savefig(image, format='png')
    image.seek(0)
    plot_image3 = base64.b64encode(image.getvalue()).decode()
    plt.close()

    response_data = {
        'plot_image1': plot_image1,
        'plot_image2': plot_image2,
        'plot_image3': plot_image3,
        'r2_score': r2,  # 위에서 계산한 R-squared 점수
        'mse': mse  # 평균제곱오차
    }

    return JsonResponse(response_data)


