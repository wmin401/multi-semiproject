# import
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import base64
from io import BytesIO
from django.http import JsonResponse

def DecisionTree(request):
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
    plt.figure(figsize=(15, 10))
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