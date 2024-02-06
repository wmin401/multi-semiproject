
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from django.http import JsonResponse
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,roc_curve, auc
import base64
from io import BytesIO
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from itertools import cycle
from sklearn import preprocessing
import matplotlib.image as mpimg

import seaborn as sns
import threading  # 새로운 라이브러리 추가
import matplotlib.font_manager as fm


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


def RandomForest(request):
    # 데이터 로딩
    file_path = 'rf_data.csv'
    train = pd.read_csv(file_path, low_memory=False)

    #인코딩, 변수지정
    X = pd.get_dummies(train.drop('여행유형', axis=1))
    y = train['여행유형']

    # 트레이닝 데이터 셋, 테스트 데이터 셋 분류
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #하이퍼파라미터 설정 및 랜덤포레스트 훈련
    param = {'n_estimators': 490, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 10, 'max_depth': None, 'criterion': 'gini', 'bootstrap': True}

    classifier = RandomForestClassifier(**param, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # 정확도 측정
    accuracy = "{:.2f}".format(metrics.accuracy_score(y_test, y_pred))
    print(f'Accuracy: {accuracy}')
    print("\n분류 보고서:\n", classification_report(y_test, y_pred))


    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1][:10]

    #한글폰트
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    font = fm.FontProperties(fname=font_path, size=10)
    plt.rc('font', family='NanumGothic')
    plt.rcParams["font.family"] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False

    # Roc곡선 그래프를 위한 인코딩
    le = preprocessing.LabelEncoder()
    y_encoded = le.fit_transform(y)

    y_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_bin.shape[1]

    # 예측 확률
    y_score = classifier.predict_proba(X_test)

    fig, axs = plt.subplots(2, 2, figsize=(13, 10))  # 충분한 subplot 생성

    # # 1. 트리 그래프
    # plot_tree(
    #     rf.estimators_[0],
    #     feature_names=X.columns,
    #     class_names=y.unique(),
    #     filled=True,
    #     ax=axs[1, 0]
    # )
    
    ## 로딩 시간 관계로 이미지로 대체 
    img = mpimg.imread('final_tree.png') 
    axs[0, 0].imshow(img)
    axs[0, 0].axis('off')

    # 2. 피쳐 중요도
    axs[0, 1].set_title('피쳐 중요도')
    axs[0, 1].bar(range(10), importances[indices], align='center')
    axs[0, 1].set_xticks(range(10))
    axs[0, 1].set_xticklabels(X.columns[indices], rotation=90)

    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f", ax=axs[1, 0])
    axs[1, 0].set_title('오차혼동행렬')
    axs[1, 0].set_xlabel('Predicted')
    axs[1, 0].set_ylabel('Actual')


    # 4. ROC
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'skyblue', 'pink'])
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        original_label = le.inverse_transform([i])
        axs[1, 1].plot(fpr, tpr, color=color, lw=2,
                    label='ROC {0} (area = {1:0.2f})'
                    ''.format(original_label, roc_auc))
        

    axs[1, 1].plot([0, 1], [0, 1], 'k--', lw=2)
    axs[1, 1].set_xlim([0.0, 1.0])
    axs[1, 1].set_ylim([0.0, 1.05])
    axs[1, 1].set_xlabel('FPR')
    axs[1, 1].set_ylabel('TPR')
    axs[1, 1].set_title('여행유형 ROC')
    axs[1, 1].legend(loc="lower right")

    plt.tight_layout()
    # 그래프를 이미지로 변환
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    # 이미지를 Base64로 인코딩
    encoded_image = base64.b64encode(image_stream.read()).decode("utf-8")
    image_stream.close()

    rf_response_data = {
        'encoded_image' : encoded_image,
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred)
    }


    return JsonResponse(rf_response_data)
