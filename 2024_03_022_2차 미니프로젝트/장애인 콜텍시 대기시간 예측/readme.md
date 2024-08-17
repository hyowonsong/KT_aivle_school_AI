🚕 장애인 콜택시 대기시간 예측(2차 미니 프로젝트)

## **📊 프로젝트 개요**

- **목적:** 장애인 이동권 개선을 위한 장애인 콜택시 대기시간 예측
- **기간:** 2024.03.22 ~ 2024.03.24
- **데이터:** 2015-01-01 ~ 2022-12-31 까지의 서울 장애인 콜택시 운행 정보
- **사용 도구:** Python, Pandas, Matplotlib, Seaborn, Scikit-learn

---

## 🚩 프로젝트 배경

- 설문조사 결과 너무 긴 대기시간(71.3%), 일정하지 않은 대기시간(10%) 이라는 문제 발생
- **콜택시 도착 예정 시간을 구해 사용자의 만족도를 올리는 것이 목표**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/36358b89-fde5-4b16-95d9-7decef74047e/f6205f5b-ab62-49ea-a088-822da3f12f39/image.png)

---

## 🖥 개발 내용

- 시계열 정형 데이터로 장애인 콜택시의 도착 예정시간을 예측하는 것이 목표
- 전처리와 데이터 분석을 진행하고 모델을 구축

### 1. 📊 데이터 전처리

```python
# 데이터 불러오기 및 병합
taxi = pd.read_csv('open_data.csv')
weather = pd.read_csv('weather.csv')
data = pd.merge(taxi, weather, on='Date')

# Feature 생성
data['Date'] = pd.to_datetime(data['Date'])
data['weekday'] = data['Date'].dt.day_name()
data['month'] = data['Date'].dt.month
data['year'] = data['Date'].dt.year
data['season'] = pd.cut(data['month'], bins=[0,3,6,9,12], labels=['Winter','Spring','Summer','Fall'])
data['avg_wait_time_7'] = data['avg_wait_time'].rolling(7).mean()

# 기초 통계 분석
print(data[['avg_wait_time', 'avg_fare', 'avg_distance']].describe())
```

- **데이터 불러오기**: `taxi_data.csv`와 `weather_data.csv`를 불러와서 분석을 위한 기본 데이터를 준비.
- **데이터 병합**: 날짜 기준으로 콜택시 데이터와 날씨 데이터를 병합하여 통합 데이터셋 구성.
- **결측치 처리**: 평균값으로 결측치를 대체하여 분석에 사용.

### 2. 🔍 데이터 탐색 및 분석

```python
# 요일별 분석
weekday_analysis = data.groupby('weekday')[['car_operation', 'avg_wait_time']].mean()
print(weekday_analysis)

# 월별 분석
monthly_analysis = data.groupby('month')[['car_operation', 'avg_wait_time']].mean()
print(monthly_analysis)

# 연도별 분석
yearly_analysis = data.groupby('year')[['car_operation', 'avg_wait_time']].mean()
print(yearly_analysis)
```

- **Feature 파생**:
    - `weekday`, `month`, `year`, `season`: 날짜로부터 요일, 월, 연도, 계절 정보를 추출.
    - `avg_waiting_last_7`: 최근 7일간의 평균 대기시간을 계산하여 추가.
    - `is_holiday`: 공휴일 정보를 추가하여 공휴일 여부를 파악.
- **상관관계 분석**:
    - 대기시간과 주요 Feature 간 상관관계를 파악.
    - 높은 상관관계를 가진 Feature가 예측에 중요한 역할을 한다는 것을 확인.

### 3. 🧠 모델링 및 평가

- **데이터 분할**:
    - 학습 데이터: 2015-01-01 ~ 2022-09-30
    - 검증 데이터: 2022-10-01 ~ 2022-12-31 (91일)
- **모델 선정 및 성능 비교**:
    - Linear Regression, SVM, Random Forest, XGBoost, LightGBM 등 다양한 모델 적용
    - 성능 지표: MAE, MAPE
- **최종 모델 성능** (LightGBM 변수 중요도 상위 7개 변수 사용, SVM 모델):
    - **MAE: 4.29**
    - **MAPE: 0.1071 (10.71%)**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/36358b89-fde5-4b16-95d9-7decef74047e/500842f7-e986-483c-9929-f913f18f7a3e/image.png)

### 4. 📈 결과 및 결론

- **평균적으로 4.29분의 오차로 대기시간을 예측할 수 있게 되었습니다.**
- **실제 대기시간의 약 10.71%의 오차 범위 내에서 예측이 가능합니다.**
