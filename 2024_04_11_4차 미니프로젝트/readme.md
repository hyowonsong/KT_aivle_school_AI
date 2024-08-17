#  🌟 쿨루프 시공 대상 여부 분류 프로젝트(4차 미니프로젝트)
---

## 📂 프로젝트 개요

- **목적** :  쿨루프와 일반 지붕을 분류하기 위한 객체 탐지 모델을 훈련
- **기간 :** 2024.04.09 ~ 2024.04.12
- **데이터** : 인공위성 이미지를 통해 추가 데이터를 수집
- **사용 도구** : YOLOv8, Python, Scikit-Learn, PyTorch, OpenCV, Google Colab, Ultralytics YOLO

---

## 🛠️ 데이터 전처리 및 준비

1. **데이터 수집 및 정리** 📊
    - 쿨루프 및 일반 지붕 이미지와 레이블 파일을 수집하여 전처리합니다.
    - 총 이미지 200장 및 레이블 200장을 확보하였습니다.
2. **데이터 스플릿** 🗂️
    - 데이터를 훈련셋과 검증셋으로 나누었습니다.
    - 훈련셋: 160장 이미지, 160장 레이블
    - 검증셋: 40장 이미지, 40장 레이블
    
    ```python
    image_paths = glob.glob("/content/drive/MyDrive/Miniproject/Datasets/2024.04.11_미니프로젝트 4차_실습자료/cool_roof_images/*.jpg")
    txt_paths = glob.glob("/content/drive/MyDrive/Miniproject/Datasets/2024.04.11_미니프로젝트 4차_실습자료/cool_roof_yolo_labels/obj_train_data/*.txt")
    
    train_images_paths, valid_images_paths, train_txt_paths, valid_txt_paths = train_test_split(image_paths, txt_paths, test_size=0.2, random_state=2024)
    
    ```
    
    - **데이터 이동**:
        
        ```python
        def file_split(train_images_paths, valid_images_paths, train_txt_paths, valid_txt_paths):
            # Train 이미지와 텍스트 파일을 이동
            for img_path, txt_path in zip(train_images_paths, train_txt_paths):
                shutil.move(img_path, data_path + 'train/images')
                shutil.move(txt_path, data_path + 'train/labels')
        
            # Valid 이미지와 텍스트 파일을 이동
            for img_path, txt_path in zip(valid_images_paths, valid_txt_paths):
                shutil.move(img_path, data_path + 'valid/images')
                shutil.move(txt_path, data_path + 'valid/labels')
        file_split(train_images_paths, valid_images_paths, train_txt_paths, valid_txt_paths)
        
        ```
        
    - **데이터 스플릿 결과**:
        - 훈련셋 이미지: 160장
        - 훈련셋 레이블: 160장
        - 검증셋 이미지: 40장
        - 검증셋 레이블: 40장

---

## 📄 YOLO 모델 설정 및 훈련

1. **YAML 파일 생성** 🔧
    - 클래스 정의 및 데이터 경로를 포함하는 YAML 파일을 생성했습니다.
    
    ```yaml
    data = {
        "names" : {0 : 'cool roof', 1 : 'generic roof'},
        "nc": 2,
        "path": data_path,
        "train": "train",
        "val": "valid",
        "test": "test",
    }
    ```
    
2. **YOLO 모델 훈련** 🚀
    - YOLOv8 모델을 사용하여 훈련을 수행했습니다.
    - 설정:
        - **모델**: [yolov8s.pt](http://yolov8s.pt/)
        - **에폭스**: 200
        - **조기 종료 인내**: 20
    
    ```python
    model = YOLO(model='yolov8s.pt', task='detect')
    model.train(data=data_path + 'data.yaml',
                epochs=200,
                patience=20,
                pretrained=True,
                verbose=True,
                seed=2024)
    ```
    
    - **훈련 결과**:
        - 최상의 모델 가중치 경로: `/content/runs/detect/train9/weights/best.pt`
    - **훈련된 모델을 사용한 추론**:
        
        ```python
        import os
        
        best_weights_path = '/content/runs/detect/train9/weights/best.pt'
        
        if os.path.exists(best_weights_path):
            model = YOLO(best_weights_path)
            results = model.predict(source=data_path + 'test/*.png',
                                    save=True,
                                    conf=0.5,
                                    iou=0.3,
                                    line_width=2)
        else:
            print(f"Error: '{best_weights_path}' 파일을 찾을 수 없습니다.")
        ```
        

---

## 🖼️ 결과 시각화

- **추론 결과 이미지 생성 코드**:
    
    ```python
    pred_path = glob.glob('/content/runs/detect/predict3/*.png')
    
    for path in pred_path:
        plt.imshow(image.load_img(path))
        plt.show()
    ```
