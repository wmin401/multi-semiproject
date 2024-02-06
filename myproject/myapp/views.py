
# myapp/views.py
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from django.http import JsonResponse
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import base64
from io import BytesIO
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression, LogisticRegression
import seaborn as sns
import urllib
import threading  # 새로운 라이브러리 추가

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

def Logistic_Regression(request):# 데이터 로딩
    # 필요한 라이브러리 import
    from io import BytesIO
    import base64
    from django.http import JsonResponse
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # 데이터 로딩
    file_path = 'LogisticRegression.csv'

    # 특정 컬럼 추출
    dataset = pd.read_csv(file_path, encoding='cp949', low_memory=False)

    # 문자열 데이터 변환: 원-핫 인코딩
    dataset_encoded = pd.get_dummies(dataset,
                                     columns=['A5_1', 'A5_2', 'A5_3', 'A5_4', 'A5_5', 'A5_6', 'A5_7', 'A5_8', 'A5_9',
                                              'A5_10', 'A5_11', 'A5_12', 'A5_13', 'A5_14', 'A5_15', 'A5_16', 'A5_17',
                                              'A5_18', 'A5_19', 'A5_20', 'A5_21'])

    # NaN 값 처리: 대체값 할당
    dataset_encoded = dataset_encoded.fillna(0)  # NaN 값을 0으로 대체

    # 출력 행 수를 230개로 설정
    pd.set_option('display.max_rows', 230)

    # 레이블 인코더를 생성합니다.
    label_encoder = LabelEncoder()

    # D_TRA1_1_SPOT 컬럼을 레이블 인코딩합니다.
    dataset['D_TRA1_1_SPOT_ENCODED'] = label_encoder.fit_transform(dataset['D_TRA1_1_SPOT'])

    # 인코딩 결과를 확인합니다.
    encoded_values = dataset[['D_TRA1_1_SPOT', 'D_TRA1_1_SPOT_ENCODED']].drop_duplicates().sort_values(
        'D_TRA1_1_SPOT_ENCODED')

    # A5_1부터 A5_21까지의 열을 원-핫 인코딩하여 새로운 열로 추가합니다.
    encoded_columns = pd.get_dummies(dataset[['A5_1', 'A5_2', 'A5_3', 'A5_4', 'A5_5', 'A5_6', 'A5_7', 'A5_8', 'A5_9',
                                              'A5_10', 'A5_11', 'A5_12', 'A5_13', 'A5_14', 'A5_15', 'A5_16', 'A5_17',
                                              'A5_18', 'A5_19', 'A5_20', 'A5_21']])

    # 새로운 데이터프레임 생성
    new_dataset = pd.concat([encoded_columns, dataset['D_TRA1_1_SPOT_ENCODED']], axis=1)

    # 컬럼명 변경을 위한 딕셔너리 생성
    column_mapping = {
        'A5_1_자연 및 풍경감상': '1',
        'A5_2_음식관광(지역 맛집 등)': '2',
        'A5_3_야외 위락 및 스포츠, 레포츠 활동': '3',
        'A5_4_역사 유적지 방문': '4',
        'A5_5_테마파크, 놀이시설, 동/식물원 방문': '5',
        'A5_6_휴식/휴양': '6',
        'A5_7_온천/스파': '7',
        'A5_7_휴식/휴양': '7',
        'A5_8_쇼핑': '8',
        'A5_9_지역 문화예술/공연/전시시설 관람': '9',
        'A5_10_스포츠 경기관람': '10',
        'A5_11_지역 축제/이벤트 참가': '11',
        'A5_12_교육/체험 프로그램 참가': '12',
        'A5_13_종교/성지순례': '13',
        'A5_14_카지노, 경마, 경륜 등': '14',
        'A5_15_시티투어': '15',
        'A5_16_드라마 촬영지 방문': '16',
        'A5_17_유흥/오락': '17',
        'A5_18_가족/친지/친구 방문': '18',
        'A5_19_회의참가/시찰': '19',
        'A5_20_교육/훈련/연수': '20',
        'A5_21_기타': '21',
        'D_TRA1_1_SPOT_ENCODED': 'travel destination'
    }

    # 컬럼명 변경
    new_dataset = new_dataset.rename(columns=column_mapping)

    # 입력 변수와 출력 변수 설정
    X = new_dataset.iloc[:, :-1]  # '1'부터 '21'까지의 열을 입력 변수로 설정
    y = new_dataset['travel destination']  # 'travel destination' 열을 출력 변수로 설정

    # 최적의 파라미터로 로지스틱 회귀 모델 생성
    model = LogisticRegression(C=1, penalty='l2')

    # 모델 훈련
    model.fit(X, y)

    # 예측
    y_pred = model.predict(X)

    # 결과 평가
    accuracy = accuracy_score(y, y_pred)

    # 예측 결과를 DataFrame으로 변환
    y_pred_df = pd.DataFrame(y_pred, columns=['Predicted Destination'])

    # 원래의 데이터셋에 예측 결과 추가
    new_dataset_with_prediction = pd.concat([new_dataset, y_pred_df], axis=1)

    # 'travel destination'의 빈도 계산
    destination_counts = new_dataset['travel destination'].value_counts()

    # 가장 인기 있는 여행지 3개 뽑기
    top_destinations = destination_counts.nlargest(4).index.tolist()

    # 229번 여행지 제외
    if 229 in top_destinations:
        top_destinations.remove(229)

    # 각 여행지에서 가장 인기 있는 여행 유형 3개 뽑기
    top_3_travel_types_per_top_destination = {}
    for destination in top_destinations:
        # 특정 여행지에 해당하는 행만 선택
        destination_data = new_dataset[new_dataset['travel destination'] == destination]

        # 여행 유형별 빈도 계산
        travel_type_counts = destination_data.iloc[:, :-2].sum()

        # 가장 빈도가 높은 여행 유형 3개 뽑기
        top_3_travel_types = travel_type_counts.nlargest(3).index.tolist()

        top_3_travel_types_per_top_destination[destination] = top_3_travel_types

    print(top_3_travel_types_per_top_destination)

    # 여행 유형과 빈도 데이터를 담을 리스트
    travel_types = []
    frequencies = []

    # 각 여행지에서 가장 인기 있는 여행 유형 3개의 빈도 계산
    for destination in top_destinations:
        destination_data = new_dataset[new_dataset['travel destination'] == destination]
        travel_type_counts = destination_data.iloc[:, :-2].sum()
        top_3_travel_types = travel_type_counts.nlargest(3)

        # 여행 유형과 빈도 데이터 추가
        travel_types.extend(top_3_travel_types.index)
        frequencies.extend(top_3_travel_types.values)

    # 막대 그래프 그리기
    x = np.arange(len(top_destinations))  # 여행지의 위치
    width = 0.2  # 막대의 너비

    fig, ax = plt.subplots()
    for i in range(3):
        if travel_types[i] == '1':
            label = 'Appreciation of Nature'
        elif travel_types[i] == '6':
            label = 'Rest/Recreation'
        elif travel_types[i] == '2':
            label = 'Visiting Good Restaurants'
        else:
            label = 'Travel Type ' + travel_types[i]

        ax.bar(x - width + i * width, frequencies[i::3], width, label=label)

    # 그래프 설정
    ax.set_xlabel('Destination')
    ax.set_ylabel('Frequency')
    ax.set_title('Top 3 Travel Types for Each Top 3 Destinations')
    ax.set_xticks(x)
    ax.set_xticklabels(['Gangneung', 'Gyeongju', 'Jeju'])
    ax.legend()

    # 그래프를 이미지 파일로 저장
    image = BytesIO()
    plt.savefig(image, format='png')
    image.seek(0)

    # 이미지 파일을 base64로 인코딩
    plot_image1 = base64.b64encode(image.getvalue()).decode()

    # 그래프 닫기
    plt.close()

    response_data = {
        'plot_image1': plot_image1,
        'accuracy': accuracy
    }

    return JsonResponse(response_data)



