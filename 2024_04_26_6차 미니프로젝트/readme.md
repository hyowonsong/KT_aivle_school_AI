# 📊 유통 판매량 예측 및 재고 최적화 프로젝트(6차 미니 프로젝트)

---

## 🎯 프로젝트 개요

- **목적:** 시계열 모델링을 통한 판매량 예측 및 재고 최적화
- **기간:** 2024.04.22 ~ 2024.04.24
- **대상 상품:**
    - Beverage (ID: 3, 가격: 8)
    - Milk (ID: 12, 가격: 6)
    - Agricultural products (ID: 42, 가격: 5)
- **사용 도구:** Python, Scikit-Learn, PyTorch, OpenCV, Keras, Statsmodels, Pandas, NumPy

---

## 🔧 데이터 준비 및 전처리

- **사용 데이터:** 판매, 주문, 유가 데이터
- **전처리 과정:**
    - 날짜 기반 특성 생성
    - 이동평균, 변화량 계산
    - 결측치 처리
    - 원-핫 인코딩

**핵심 코드:**

```python
def preprocessing4predict(sales_test, orders_test, oil_price_test, products, stores, Product_ID):
    sales_44 = sales_test[sales_test['Store_ID']==44]
    sales_44 = sales_44[sales_44['Product_ID']==Product_ID]
    sales_44['week'] = sales_44['Date'].dt.day_name()
    sales_44['month'] = sales_44['Date'].dt.month
    sales_44 = pd.merge(sales_44, orders_test, how='left')
    sales_44 = pd.merge(sales_44, oil_price_test, how='left')
    sales_44['Qty_mean_7'] = sales_44['Qty'].rolling(7, min_periods=1).mean()
    sales_44['Qty_diff'] = sales_44['Qty'].shift(-2) - sales_44['Qty']
    sales_44['Qty_2day'] = sales_44['Qty'].shift(-2)
    # ... (추가 전처리 단계)
    return x_test, y_test, scaler, y_min, y_max
```

---

## 🤖 모델링

- **사용 모델:** LSTM
- **모델 구조:**
    - **LSTM:** 128 → 64 → 32 → 16 → 8 → 1

**LSTM 모델 핵심 코드:**

```python
model = Sequential([
    LSTM(128, input_shape=(ts, nfeat), activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
```

---

## 📏 평가 지표

- RMSE, MAE, MAPE, R2

---

## 📈 결과

1. **🥤 Beverage (ID: 3)**
    - **LSTM 성능:**
        - RMSE: 4383.31
        - MAE: 4296.94
        - MAPE: 0.4376
        - R2: -0.0920
    - **재고 시뮬레이션:**
        - 일평균 재고량: 10039.286
        - 일평균 재고 금액: 80314.288
        - 일평균 재고회전율: 1.213
        - 기회손실 수량: 0.0
2. **🥛 Milk (ID: 12)**
    - **LSTM 성능:**
        - RMSE: 852.44
        - MAE: 677.63
        - MAPE: 0.0694
        - R2: 0.9334
    - **재고 시뮬레이션:**
        - 일평균 재고량: 12503.69
        - 일평균 재고 금액: 75022.14
        - 일평균 재고회전율: 0.949
        - 기회손실 수량: 0.0
3. **🌾 Agricultural products (ID: 42)**
    - **LSTM 성능:**
        - RMSE: 9.47
        - MAE: 8.27
        - MAPE: 0.0742
        - R2: 0.6354
    - **재고 시뮬레이션:**
        - 일평균 재고량: 127.238
        - 일평균 재고 금액: 636.19
        - 일평균 재고회전율: 0.978
        - 기회손실 수량: 0.0

**재고 시뮬레이션 핵심 코드:**

```python
def inv_simulator(y, pred, safe_stock, price):
    temp = pd.DataFrame({'y': y.reshape(-1,), 'pred': pred.reshape(-1,).round()})
    temp['base_stock'] = temp['close_stock'] = temp['order'] = temp['receive'] = 0

    for i in range(len(temp)-2):
        # ... (재고 계산 로직)

    DailyStock = ((inventory['base_stock'] + inventory['close_stock'])/2)
    DailyTurnover = (inventory['y'] + inventory['lost']) / DailyStock

    AvgDailyStock = round(DailyStock.mean(), 3)
    AvgDailyStockAmt = AvgDailyStock * price
    turnover = round(DailyTurnover.mean(), 3)
    lost_sum = inventory['lost'].sum()

    # ... (결과 출력)

    return inventory
```

---

## 🎓 결론

### LSTM 성능 평가

- **Milk** 제품의 예측 성능이 가장 우수함 (R2: 0.9334)
- **Agricultural products**가 두 번째로 좋은 성능을 보임 (R2: 0.6354)
- **Beverage** 제품의 예측 성능이 부족함 (R2: -0.0920)

### 재고 시뮬레이션

- 모든 제품에서 기회손실 수량이 0으로 달성됨
- **Beverage**의 재고회전율이 가장 높음 (1.213)
- **Agricultural products**의 일평균 재고 금액이 가장 낮음 (636.19)
