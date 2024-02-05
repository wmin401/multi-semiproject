
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
import threading  # 새로운 라이브러리 추가

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from django.http import JsonResponse
from django.views import View
import base64
from io import BytesIO


class SupportVectorMachine(View):
    def get(self, request):
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

        loaded_model = joblib.load('svm_model.pkl')
        predictions = loaded_model.predict(X_test)
        accuracy = accuracy_score(Y_test, predictions)

        with open('xx.pkl', 'rb') as f:
            xx = pickle.load(f)

        with open('yy.pkl', 'rb') as f:
            yy = pickle.load(f)

        with open('Z.pkl', 'rb') as f:
            Z = pickle.load(f)

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
        ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=Y_train, s=20, edgecolor='k', cmap=plt.cm.coolwarm)
        ax.set_xlabel('Total transportation cost')
        ax.set_ylabel('Number_of_people')
        ax.set_title('SVM Decision Boundary with Best Parameters')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        buf = BytesIO()
        plt.savefig(buf, format='png')
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
        buf.close()

        return JsonResponse({
            'predictions': predictions.tolist(),
            'accuracy': accuracy,
            'graph': image_base64
        })
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

