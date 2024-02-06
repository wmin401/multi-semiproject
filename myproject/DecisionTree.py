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

# CSV 파일 경로
d = 'COST_and_ONECOST.csv'

print("==== 전처리 시작 ====")
# CSV 파일 읽고, 인코딩 설정
df = pd.read_csv(d, encoding='cp949', low_memory=False)

# 필요한 컬럼 추출 (COST = 여행 총 경비 / ONE_COST = 1인 지출 비용)
df1 = df[['D_TRA1_COST', 'D_TRA1_ONE_COST']].copy()


# 결측치 처리 둘중에 선택 전자는 평균값으로 / 후자는 결측치행삭제 / 전처리작업
#df1.fillna(df1.mean(), inplace=True)
df1.dropna(inplace=True)
print(df1.isnull().sum())
print("==== 전처리 끝 ====")

# 데이터 스케일링
scaler = StandardScaler()
df1_scaled = pd.DataFrame(scaler.fit_transform(df1), columns=df1.columns)

# 데이터프레임 내용 확인
print("==== 데이터 프레임 내용 확인 시작 ====")
print(df1_scaled)
print("==== 데이터 프레임 내용 확인 끝 ====")

# EDA 작업
print("==== EDA 시작 ====")
print(df1_scaled.head())
print(df1_scaled.info())
print(df1_scaled.describe())
print(df1_scaled.isnull().sum())
# 각 변수 분포 확인
print("=== 변수 분포 스케터 ===")
plt.figure(figsize=(15, 10))
plt.scatter(df1_scaled['D_TRA1_COST'], df1_scaled['D_TRA1_ONE_COST'])
# title
plt.title("Scatter plot of Variable Distribution")
# Axis Labels
plt.xlabel("COST")
plt.ylabel("People")
plt.xticks([])
plt.yticks([])
plt.show()
# 토탈 상관계수
print("=== 상관계수 ===")
sl_col = df1_scaled[['D_TRA1_COST', 'D_TRA1_ONE_COST']]
sns.heatmap(sl_col.corr(), annot=True, fmt=".2f")
plt.show()
print("==== EDA 끝 ====")

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
print("==== 최적의 파라미터와 점수 출력 시작 ====")
print("Best parameters: ", rand_search.best_params_)
print("Best cross-validation score: ", -rand_search.best_score_)
print("==== 최적의 파라미터와 점수 출력 끝 ====")

# 최적 하이퍼 파라미터로 학습된 모델 가져오기
DTR = rand_search.best_estimator_

# 테스트 데이터로 예측
y_pred = DTR.predict(X_test)

# 특성 중요도
print("==== 특성 중요도 확인 시작 ====")
print("Feature importances :", DTR.feature_importances_)
print("==== 특성 중요도 확인 끝 ====")

# 모델 성능 평가
print("==== 모델 성능 평가 시작 ====")
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)
# MAE 계산(예측 정확도)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error: ", mae)
# R-Squared 계산(결정계수)
r2 = r2_score(y_test, y_pred)
print("R-squared: ", r2)
print("==== 모델 성능 평가 끝 ====")

# 시각화
print("==== 의사결정트리 시각화 시작 ====")
plt.figure(figsize=(10, 7))
tree.plot_tree(DTR, max_depth=4, feature_names=X.columns, filled=True)
plt.title("Decision Tree Visualization")
plt.show()
print("==== 의사결정트리 시각화 끝 ====")