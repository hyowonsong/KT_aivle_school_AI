# 📱 **스마트폰 센서 데이터 기반 모션 분류 프로젝트(3차 미니 프로젝트)**

---

## **📊 프로젝트 개요**

- **목적:** 스마트폰 센서 데이터를 활용해 사용자의 모션을 분류하는 모델을 구축
- **기간:** 2024.04.04 ~ 2024.04.08
- **데이터:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- **사용 도구:** Python, Pandas, Matplotlib, Seaborn, Scikit-learn

---

## 📑 **프로젝트 단계**

### **단계 1**: 정적(0), 동적(1) 행동 분류 모델 생성

- **목표**: 사용자의 행동을 정적과 동적으로 분류
- **모델링 방법**: 여러 알고리즘을 활용해 모델링을 수행하고, 성능이 가장 좋은 모델을 선정

### **단계 2**: 세부 동작 분류 모델 생성

- **목표**: 단계 1에서 예측한 행동을 기준으로, 세부 동작을 추가적으로 분류
    - **정적 행동 분류**: Laying, Sitting, Standing
    - **동적 행동 분류**: Walking, Walking Upstairs, Walking Downstairs
- **모델링 방법**: 여러 알고리즘을 활용해 각 동작을 분류하고, 성능이 가장 좋은 모델을 선정

### **모델 통합**

- **목표**: 단계별로 나눈 모델을 통합해 최종 예측 결과를 도출
- **성능 평가**: 통합된 모델의 예측 성능을 평가하고, 최적의 성능을 달성

---

## 🛠️ **환경 설정**

### 1️⃣ **필요 라이브러리**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import re
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
```

### 2️⃣ **데이터 불러오기**

```python
data = pd.read_csv('data01_train.csv')
new_data = pd.read_csv('data01_test.csv')
data = data.drop('subject', axis = 1)
data.head()
```

---

## 🧹 **데이터 전처리**

### 1️⃣ **Label 추가 및 데이터 분할**

- **Activity_dynamic**: 정적(0) 또는 동적(1) 여부를 나타내는 라벨을 추가
- **Train/Validation Split**: `train`과 `val` 데이터를 8:2 로 분할

```python
data['Activity_dynamic'] = np.where(data['Activity'].isin(['STANDING', 'SITTING', 'LAYING']), 0, 1)
X = data.drop(['Activity', 'Activity_dynamic'], axis = 1)
y1 = data['Activity']
y2 = data['Activity_dynamic']
X_train, X_val, y_train, y_val = train_test_split(X, y1, test_size = 0.2, random_state = 42)
X_train2, X_val2, y_train2, y_val2 = train_test_split(X, y2, test_size = 0.2, random_state = 42)
```

---

## 📊 **단계별 모델링**

### **🔍 단계 1: 정적/동적 행동 분류 모델**

- **알고리즘**: Random Forest
- **모델링 결과**: 높은 정확도를 가진 Random Forest 모델을 선정

```python
model_rf = RandomForestClassifier(random_state = 42)
model_rf.fit(X_train2, y_train2)
p_rf = model_rf.predict(X_val2)
```

- **모델 성능 평가**
    - **Accuracy**: 98%
    - **Confusion Matrix** 및 **Classification Report** 출력.
    

### **🔍 단계 2-1: 정적 동작 세부 분류**

- **알고리즘**: CatBoost
- **특성 중요도 분석**: 상위 25개 중요한 특성을 선정하여 최적 모델 구축.

```python
model_cat = CatBoostClassifier(random_state = 42, task_type = 'GPU', verbose = 0)
model_cat.fit(X_train_25, y_train3)
p_cat = model_cat.predict(X_val_25)
```

- **모델 성능 평가**
    - **Accuracy**: 99%
    - **Confusion Matrix** 및 **Classification Report** 출력.

### **🔍 단계 2-2: 동적 동작 세부 분류**

- **알고리즘**: CatBoost
- **모델링 결과**: CatBoost를 사용하여 동적 동작 세부 분류를 수행

```python
model_cat = CatBoostClassifier(random_state = 42, task_type = 'GPU', verbose = 0)
model_cat.fit(X_train4, y_train4)
p_cat = model_cat.predict(X_val4)
```

- **모델 성능 평가**
    - **Accuracy**: 99%
    - **Confusion Matrix** 및 **Classification Report** 출력.

---

## 🧠 **모델 통합 및 성능 평가**

### **함수 구현 및 모델 통합**

- **함수**: 새로운 데이터에 대해 전체 파이프라인을 통해 예측을 수행하고, 성능을 평가하는 함수 작성.

```python
def predict_new_data(path):
    # 데이터 로드 및 전처리
    new_data = pd.read_csv(path)
    new_data.drop('subject', axis=1, inplace=True)
    X_test = new_data.drop('Activity', axis = 1)
    y_test = new_data['Activity']

    # 모델 로드
    model1 = joblib.load('model1.pkl')
    model2_1 = joblib.load('model2_1.pkl')
    model2_1_25 = joblib.load('model2_1_25.pkl')
    model2_2 = joblib.load('model2_2.pkl')
    top_25_feature = joblib.load('cat_top_25_features.pkl')

    # 동적 정적 유무 예측 및 세부 예측
    pred1= model1.predict(X_test)
    # 예측 결과 합치기 및 성능 평가
    print('<model2_1 all features>\\n')
    # 평가 결과 출력
```

- **성능 평가 결과**
    - **All Features**: Accuracy 99%
    - **Top 25 Features**: Accuracy 99%

---

## 🔍 **결론**

- 단계별 모델링을 통해 높은 성능의 예측 모델을 구축
- 전체 시스템을 통합하여 일관된 성능을 유지
- 결과적으로,  **다양한 스마트폰 센서 데이터를 기반으로 사용자의 활동을 효과적으로 분류**
