
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
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import base64
from io import BytesIO
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression
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
