
### 모델 학습 및 학습된 모델 테스트 실행 코드

'''
학습 데이터
첫번째 데이터 셋 : 감성대화 말뭉치 + 한국어 단발성 대화 데이터셋
감정 매핑 : 행복 놀람 분노 슬픔 불안

두번째 데이터 셋 : 감성대화 말뭉치 (감정 대분류 , 사용자 텍스트 + 감정소분류(세분화 된 감정))
감정분류 : 공포 놀람 분노 슬픔 중립 행복 혐오
'''

# 필요한 패키지 설치
# !pip install mxnet
# !pip install gluonnlp pandas tqdm
# !pip install sentencepiece
# !pip install transformers==4.8.1
# !pip install torch==1.10.1
# !pip install git+https://git@github.com/SKTBrain/KoBERT.git@master

## 실제 gpu 학습 및 테스트
# 패키지 임포트
import torch
import os
import sys
import io
from torch import nn
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import pandas as pd
import numpy as np
from tqdm import tqdm
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# KoBERT 모델 및 Vocabulary 로드
bert_model, vocab = get_pytorch_kobert_model()
tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)

# 데이터 로드 및 전처리
# 데이터셋 로드
chatbot_data = pd.read_csv('../../data_set/all_emotions_oversampled.csv')

# 결측치 제거
chatbot_data = chatbot_data.dropna(subset=['text', 'emotion'])

# 감정 레이블 매핑
label_dict = {"분노": 0, "행복": 1, "불안": 2, "놀람": 3, "슬픔": 4}
chatbot_data['emotion'] = chatbot_data['emotion'].map(label_dict)

# 데이터 불균형 확인
print(chatbot_data['emotion'].value_counts())

# 데이터 리스트로 변환
data_list = chatbot_data[['text', 'emotion']].values.tolist()

# 데이터셋 분리
train_data, test_data = train_test_split(data_list, test_size=0.25, random_state=42, stratify=chatbot_data['emotion'])

# 데이터셋 클래스 정의
class BERTDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len=128):
        self.data = dataset
        self.transform = nlp.data.BERTSentenceTransform(
            tokenizer, max_seq_length=max_len, pad=True, pair=False
        )

    def __getitem__(self, idx):
        text = self.data[idx][0]
        label = int(self.data[idx][1])
        input_ids, valid_length, token_type_ids = self.transform([text])
        return (
            torch.tensor(input_ids),
            torch.tensor(valid_length),
            torch.tensor(token_type_ids),
            torch.tensor(label)
        )

    def __len__(self):
        return len(self.data)

# 데이터셋 생성
batch_size = 16
max_len = 128

train_dataset = BERTDataset(train_data, tokenizer, max_len)
test_dataset = BERTDataset(test_data, tokenizer, max_len)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 정의
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=5, dr_rate=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(p=dr_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        cls_token = outputs[0][:, 0]  # [CLS] 토큰
        pooled_output = self.dropout(cls_token)
        return self.classifier(pooled_output)

# GPU 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_per_process_memory_fraction(0.3, device)
model = BERTClassifier(bert_model, dr_rate=0.3).to(device)

# 학습 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

# 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        input_ids, valid_length, token_type_ids, labels = [item.to(device) for item in batch]
        
        # Attention Mask 수정
        attention_mask = (input_ids != 0).long()

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(train_dataloader):.4f}")

# 모델 저장
model_save_path = "kobert_emotion_model_fixed.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# 평가 함수
def evaluate_model(model, dataloader, device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, valid_length, token_type_ids, labels = [item.to(device) for item in batch]
            attention_mask = (input_ids != 0).long()
            outputs = model(input_ids, attention_mask, token_type_ids)
            preds = torch.argmax(outputs, dim=-1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    acc = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='weighted')
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')
    return acc, f1, precision, recall

# 모델 평가
acc, f1, precision, recall = evaluate_model(model, test_dataloader, device)
print(f"Test Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

