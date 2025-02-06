# Uni_OSS
오픈소스소프트웨어 수업

---

# **README.md: KoBERT 기반 감정 분류 서비스중 AI 모델구현**

---

## 📌 **프로젝트 개요**

이 프로젝트는 **SKTBrain의 KoBERT** 모델을 활용해 **한국어 텍스트 감정 분류**를 수행합니다.  
**주요 기능**:
1. **데이터 전처리**: AI Hub 데이터셋 병합 및 균형화
2. **모델 학습**: GPU를 활용한 KoBERT 모델 학습
3. **실시간 감정 예측**: 감정 분류와 확률 표시
4. **다양한 실행 환경**: GPU와 CPU 테스트 지원

---

## 📂 **프로젝트 구조**

```plaintext
project/
├── data_set/                           # 데이터셋 저장 폴더 (merge_data.py, modification_data.py  실행시 생성)
│   ├── all_emotions_combined.csv       # 병합된 감성대화 데이터
│   ├── all_emotions_oversampled.csv    # 오버샘플링된 데이터 
│   ├── emotion_merge_subemotion_text.csv
│   ├── emotion_two_columns.csv
│   ├── sub_emotion_two_columns.csv
│   └── korea_two_columns.csv
├── non_classified_data/                # 전처리 이전 데이터
│   ├── korea_list.xlsx                 # 한국어 단발성 대화(AI허브)
│   └── emotion_list.xlsx               # 감성 대화 말뭉치(AI허브)
├── requirements.txt                    # 프로젝트 의존성 패키지 목록
├── readME.md                           # 프로젝트 설명서
├── LICENSE                             
└── scripts/                            # 스크립트 폴더
    ├── preprocessing_data/             # 데이터 전처리 관련 코드
    │   ├── merge_data.py               # 데이터 처리 및 병합
    │   └── modification_data.py        # 데이터 갯수 고유값 확인 및 오버 샘플링
    ├── run_model/                      # 모델 실행 관련 코드
    │   ├── run_model.py                      # 최종 감정만 출력
    │   └── run_percentage_emotion_mode.py    # 최종 감정과 각 확률 출력(최종감정: 최고확률 감정)
    ├── train_model/                    # 모델 학습 관련 코드
    │   ├── train_model.py              
    │   └── test_model_cpu.py           
```

---

## 🛠️ **설치 및 실행 방법**

### **1. 환경 설정**

#### **Python 환경 설정**
- **Python 버전**: 3.8 이상
- **CUDA 버전**: 12.3  
- **NVIDIA GPU**: RTX A6000 기준 (지원 확인 완료)

#### **필수 패키지 설치**
```bash
pip install -r requirements.txt
```

`requirements.txt`:
```plaintext
torch==1.10.1
transformers==4.8.1
gluonnlp==0.10.0
sentencepiece
mxnet
kobert
pandas
tqdm
numpy
```

---

### **2. 데이터 출처**

본 프로젝트에서 사용된 데이터는 **AI Hub**에서 제공한 **공개 데이터셋**을 활용하였습니다:

1. **감성대화 말뭉치**  
   - **설명**: 다양한 감정과 상황에 기반한 대화 데이터를 제공  
   - **출처**: [AI Hub 감성대화 말뭉치](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=86)
   - **주요 컬럼**:
     - `감정 대분류`: 기쁨, 분노, 불안, 슬픔 등
     - `감정 소분류`: 세부 감정 레이블
     - `사람문장1`, `사람문장2`, `사람문장3`: 대화 텍스트 데이터

2. **한국어 단발성 대화 데이터셋**  
   - **설명**: 단일 문장을 바탕으로 감정을 분류한 대화 데이터  
   - **출처**: [AI Hub 한국어 단발성 대화](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=270)
   - **주요 컬럼**:
     - `Emotion`: 감정 레이블 (행복, 놀람, 분노, 슬픔, 불안 등)
     - `Sentence`: 문장 데이터

> **출처 명시**: 본 데이터는 **AI Hub**를 통해 제공받았습니다.  
> 출처: [AI Hub (https://www.aihub.or.kr)](https://www.aihub.or.kr)

---

### **3. 데이터 전처리**

데이터 병합 및 감정 레이블링을 수행합니다:
```bash
python scripts/preprocessing_data/merge_data.py
python scripts/preprocessing_data/modification_data.py
```

**결과 파일**:
- `data_set/all_emotions_combined.csv`
- `data_set/all_emotions_balanced.csv`

---

### **4. KoBERT 모델 학습**

GPU를 활용해 KoBERT 모델을 학습합니다:
```bash
python scripts/train_model/train_model.py
```

**학습 결과**:
- 학습된 모델 저장: `kobert_emotion_model_fixed.pth(임의의이름 지정)`

### 🛠️ **학습 파라미터**

| **파라미터**            | **설정 값**           |
|-------------------------|-----------------------|
| **모델**                | KoBERT               |
| **드롭아웃 비율**       | 0.3                  |
| **Hidden Size**         | 768                  |
| **클래스 수**           | 5                    |
| **학습률 (Learning Rate)** | 3e-5                 |
| **손실 함수**           | CrossEntropyLoss     |
| **배치 크기 (Batch Size)**| 16                   |
| **Epoch 수**            | 10                   |
| **최대 시퀀스 길이**     | 128                  |
| **Train/Test Split**    | 75% / 25%            |
| **GPU 메모리 제한**     | 30%                  |

---

### 📊 **모델 성능 평가 결과**

| **평가지표**       | **값**      |
|--------------------|-------------|
| **Test Accuracy**  | 0.8797      |
| **F1 Score**       | 0.8796      |
| **Precision**      | 0.8800      |
| **Recall**         | 0.8797      |


---

### **5. 실시간 감정 예측**

#### **GPU 기반 실시간 감정 예측 (확률 포함)**  
```bash
python scripts/run_model/run_percentage_emotion_mode.py
```

**실행 예시**:
```plaintext
문장을 입력하세요 (종료하려면 'exit' 입력): 오늘 날씨가 너무 좋아서 기분이 좋아요!
입력 문장: 오늘 날씨가 너무 좋아서 기분이 좋아요!
예측된 감정: 행복
감정별 확률:
 - 분노: 1.23%
 - 행복: 89.34%
 - 불안: 2.12%
 - 놀람: 4.56%
 - 슬픔: 2.75%
```

---

## 📊 **결과물**

1. **학습된 모델**: `kobert_emotion_model_fixed.pth(임의의 이름지정)`
2. **데이터셋**: AI Hub 데이터셋 병합 및 균형화 파일
3. **실행 결과**:
   - 감정 예측 및 감정별 확률 출력

---

## 📄 **라이센스**

본 프로젝트는 [KoBERT](https://github.com/SKTBrain/KoBERT)의 **Apache License 2.0**를 따릅니다.  
또한 AI Hub의 데이터를 사용하였으며, AI Hub의 이용약관과 저작권을 준수합니다.  

- **데이터 출처**: [AI Hub (https://www.aihub.or.kr)](https://www.aihub.or.kr)
- **KoBERT 라이센스**: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## 📧 **문의**

- **작성자**: 손영준
- **이메일**: dragondall2015@gmail.com

---
