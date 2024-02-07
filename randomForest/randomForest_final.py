from itertools import cycle
from lightgbm import plot_tree
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.calibration import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns


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
fig, axs = plt.subplots(figsize=(13, 10))  # 충분한 subplot 생성

 # 1. 트리 그래프
plot_tree(
    classifier.estimators_[0],
    feature_names=X.columns,
    class_names=y.unique(),
    filled=True,
    ax=axs[0, 0]
)
# 2. 피쳐 중요도
axs[0, 1].set_title('피쳐 중요도')
axs[0, 1].bar(range(10), importances[indices], align='center')
axs[0, 1].set_xticks(range(10))
axs[0, 1].set_xticklabels(X.columns[indices], rotation=90)
# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt=".0f", ax=axs[1, 0])
axs[1, 0].set_title('Confusion Matrix')
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
axs[1, 1].set_title('ROC')
axs[1, 1].legend(loc="lower right")
plt.tight_layout()

plt.show()
