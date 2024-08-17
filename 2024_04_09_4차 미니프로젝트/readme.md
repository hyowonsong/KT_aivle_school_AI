# 🚗 차량 공유업체의 차량 파손 여부 분류하기(4차미니프로젝트)
---

## **📊 프로젝트 개요**

- **목적:** 스마트폰 센서 데이터를 활용해 사용자의 모션을 분류하는 모델을 구축
- **기간:** 2024.04.09 ~ 2024.04.12
- **데이터:** **Car_Images.zip**: 차량의 정상/파손 이미지 무작위 수집
- **사용 도구:** Python, Pandas, Matplotlib, Seaborn, Scikit-learn

---

### 🧹 **데이터 전처리**

### 과제 수행 목표

- **데이터를 모델링에 적합한 형태로 정리하기**

### 작업 단계 및 코드

1. **데이터 전처리**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_data(image_paths, label):
    data = []
    labels = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=(128, 128))
        img = img_to_array(img)
        data.append(img)
        labels.append(label)
    return np.array(data), np.array(labels)

normal_paths = [f'/content/car_images/normal/{img}' for img in normal_images]
damaged_paths = [f'/content/car_images/damaged/{img}' for img in damaged_images]

X_normal, y_normal = load_data(normal_paths, 0)
X_damaged, y_damaged = load_data(damaged_paths, 1)

X = np.concatenate([X_normal, X_damaged])
y = np.concatenate([y_normal, y_damaged])

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.1, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

```

---

### 🤖 미션 2: CNN 모델링

### 과제 수행 목표

- **Keras를 이용하여 3개 이상의 모델 생성 및 성능 비교**

### 모델 1: 기본 CNN

- **훈련 정확도**: 87.2%
- **검증 정확도**: 84.5%
- **훈련 손실**: 0.35
- **검증 손실**: 0.42

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

model1 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

history1 = model1.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), callbacks=[early_stopping])

```

### 모델 2: Learning Rate 조정

- **훈련 정확도**: 89.1%
- **검증 정확도**: 86.3%
- **훈련 손실**: 0.30
- **검증 손실**: 0.37

```python
from tensorflow.keras.optimizers import Adam

model2 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.0005)
model2.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
history2 = model2.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), callbacks=[early_stopping])

```

### 모델 3: Deep CNN with Dropout

- **훈련 정확도**: 90.4%
- **검증 정확도**: 88.2%
- **훈련 손실**: 0.28
- **검증 손실**: 0.33

```python
from tensorflow.keras.layers import Dropout

model3 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.0001)
model3.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
history3 = model3.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), callbacks=[early_stopping])

```

---

### 🔄 미션 3: Transfer Learning

### 과제 수행 목표

- **성능 개선을 위해 두 가지 시도**

### 작업 단계 및 코드

1. **Image Preprocessing Layer & Image Augmentation Layer**

```python
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)
```

1. **Transfer Learning (Inception V3)**
- **훈련 정확도**: 93.6%
- **검증 정확도**: 91.2%
- **훈련 손실**: 0.22
- **검증 손실**: 0.29

```python
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model_transfer = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model_transfer.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_transfer = model_transfer.fit(datagen.flow(X_train, y_train, batch_size=32),
                                       epochs=20,
                                       validation_data=(X_val, y_val),
                                       callbacks=[early_stopping])
plot_history(history_transfer, 'Transfer Learning Model')
```

## 🔍 **결론**

- 결과를 종합해 보면, **Transfer Learning**을 적용한 모델이 가장 우수한 성능을 보여주고 있습니다.
- 따라서 **Transfer Learning**을 적용한 모델이 가장 좋은 선택입니다.

### 성능 요약

| 모델 | 훈련 정확도 | 검증 정확도 | 훈련 손실 | 검증 손실 |
| --- | --- | --- | --- | --- |
| 기본 CNN | 87.2% | 84.5% | 0.35 | 0.42 |
| Learning Rate 조정 | 89.1% | 86.3% | 0.30 | 0.37 |
| Deep CNN with Dropout | 90.4% | 88.2% | 0.28 | 0.33 |
| Transfer Learning | 93.6% | 91.2% | 0.22 | 0.29 |
