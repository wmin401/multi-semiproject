
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
import base64
from io import BytesIO
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression, LogisticRegression
import seaborn as sns
import urllib
import threading  # 새로운 라이브러리 추가
import pickle
import joblib
from sklearn import svm



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



