# 🚴‍♀️ 서울시 공유자전거 '따릉이' 수요 예측 프로젝트

---

### **📊 프로젝트 개요**

- **목적:** 서울시 공유자전거 '따릉이'의 시간대별, 요일별 수요 패턴 분석
- **기간:** 2024.03.04 ~ 2024.03.06
- **데이터:** 서울시 공공데이터 (5,827개 샘플, 12개 변수)
- **사용 도구:** Python, Pandas, Matplotlib, Seaborn

---

### **🌆 프로젝트 배경**

- 서울시는 '따릉이' 서비스의 **효율적 운영**을 위해 시간대별, 요일별 수요 패턴 분석이 필요
- 이를 통해 자전거 재배치 전략 수립 및 운영 최적화를 목표

---

### **💻 개발 과정 및 분석 결과**

### **1. 데이터 전처리 및 기본 분석**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로딩
df = pd.read_csv('df_sbikedata.csv')

# 날짜 데이터 처리
df['date'] = pd.to_datetime(df['date'])
df['weekday'] = df['date'].dt.weekday
df['weekend'] = np.where(df['weekday'] >= 5, 1, 0)

# 시간대 그룹화
bins = [0, 6, 12, 18, 24]
labels = ['0-6시', '6-12시', '12-18시', '18-24시']
df['time_group'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False)
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/36358b89-fde5-4b16-95d9-7decef74047e/5b809ba2-0959-4305-b61e-3706a7ddcc65/image.png)

---

### **2. 시간대별 분석**

```python
# 시간대별 평균 대여량 계산
result_df = df.groupby('time_group')['count'].mean()

# 시각화
plt.figure(figsize=(10, 6))
result_df.plot(kind='bar', color='skyblue')
plt.title('시간대별 평균 따릉이 대여량')
plt.xticks(rotation=45)
plt.show()
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/36358b89-fde5-4b16-95d9-7decef74047e/44ec5bd2-3831-49a0-a421-f07541200586/image.png)

**결과:** 

- **18-24시** 구간의 평균 대여량이 가장 높으며, **0-6시** 구간이 가장 낮습니다. (**7배** 높은 대여량)

---

### **3. 주말/평일 분석**

```python
# 주말과 평일의 시간대별 평균 대여량 계산
result_df_weekday = df[df['weekend'] == 0].groupby('time_group')['count'].mean()
result_df_weekend = df[df['weekend'] == 1].groupby('time_group')['count'].mean()

# 시각화
plt.figure(figsize=(10, 6))
result_df_weekday.plot(label='평일')
result_df_weekend.plot(label='주말')
plt.title('주말과 평일 시간대별 평균 따릉이 대여량')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# 총 대여량 비교
total_rental_weekday = df[df['weekend'] == 0]['count'].sum()
total_rental_weekend = df[df['weekend'] == 1]['count'].sum()
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/36358b89-fde5-4b16-95d9-7decef74047e/db7cef81-e35a-444d-80fc-ba2138996426/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/36358b89-fde5-4b16-95d9-7decef74047e/bb2af25f-0b5c-4d58-8d6c-71f738895b26/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/36358b89-fde5-4b16-95d9-7decef74047e/2e35dbce-9bd7-4569-985a-c4381d124d47/image.png)

**결과:**

- 평일이 주말보다 **높은 대여량**

**분석:**

1. **평일**은 출퇴근 시간대(6-12시, 18-24시)에 뚜렷한 피크를 보입니다.
2. **주말**은 12-18시에 가장 높은 대여량을 보이며, 시간대별 변동이 평일보다 완만합니다.
3. **평일 총 대여량이 주말보다 현저히 높아**, 평일 중심의 운영 전략이 필요함을 시사합니다.

---

### **📈 정책 제언**

1. **시간대별 자전거 재배치**
    - **12-18시** 구간에 더 많은 자전거 배치
    - **예상 효과:** 피크 시간 대여 가능성 **증가**
2. **평일 출퇴근 시간 특별 운영**
    - **6-9시, 18-21시**에 주요 대여소 인력 배치 강화
    - **예상 효과:** 사용자 만족도 **상승**, 대여량 **증가**
3. **심야 시간대 (0-6시) 운영 최적화**
    - 심야 시간 일부 대여소 운영 중단, 자전거 정비에 활용
    - **예상 효과:** 운영 비용 **절감**, 자전거 가용률 **향상**
