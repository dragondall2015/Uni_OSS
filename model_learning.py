# 패키지 임포트
import torch
import os
import sys
import io
from torch import nn
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import pandas as pd
from tqdm import tqdm
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from sklearn.model_selection import train_test_split

# KoBERT 모델 및 Vocabulary 로드
bert_model, vocab = get_pytorch_kobert_model()
tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)

# 데이터 로드 및 전처리
# 첫번째 데이터셋 위치
# chatbot_data = pd.read_csv('./data_set/all_emotions_combined.csv')
# 두번째 데이터셋 위치
# chatbot_data = pd.read_csv('./data_set/emotion_merge_subemotion_text.csv')

# 감정 레이블 매핑
# 첫번째 데이터 셋 레이블 
label_dict = {"분노": 0, "행복": 1, "불안": 2, "놀람": 3, "슬픔": 4}
# 두번째 데이터 셋 레이블 상처와 슬픔을 하나로 묶음
# label_dict = {"기쁨": 0, "당황": 1, "분노": 2, "불안": 3, "상처": 4, "슬픔": 4}
chatbot_data['emotion'] = chatbot_data['emotion'].map(label_dict)
data_list = chatbot_data[['text', 'emotion']].values.tolist()

# 데이터셋 분리
train_data, test_data = train_test_split(data_list, test_size=0.25, random_state=42)

# 데이터셋 클래스 정의
class BERTDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len=64):
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
batch_size = 32
max_len = 128

train_dataset = BERTDataset(train_data, tokenizer, max_len)
test_dataset = BERTDataset(test_data, tokenizer, max_len)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 정의
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=5, dr_rate=0.6):
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

# GPU 설정 (GPU 0번 사용, 메모리 사용량 30%로 제한)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_per_process_memory_fraction(0.3, device)
model = BERTClassifier(bert_model, dr_rate=0.6).to(device)

# 모델 초기화
model = BERTClassifier(bert_model, dr_rate=0.8).to(device)

# 첫번째 모델 path
# model_save_path = "kobert_emotion_model.pth"
model_save_path = "hyper2_kobert_emotion_model.pth"
# 두번재 모델 path
# model_save_path = "kobert_emotion_model_2.pth"

# 모델 학습 또는 로드
if os.path.exists(model_save_path):
    print("학습된 모델이 존재합니다. 모델을 로드합니다.")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
else:
    print("모델이 없습니다. 학습을 시작합니다.")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids, valid_length, token_type_ids, labels = [item.to(device) for item in batch]
            attention_mask = torch.arange(input_ids.size(1)).expand(len(valid_length), input_ids.size(1)).to(device) < valid_length.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(train_dataloader):.4f}")
    # 모델 저장
    torch.save(model.state_dict(), model_save_path)
    print(f"모델이 저장되었습니다: {model_save_path}")

# 평가 함수
def calc_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, valid_length, token_type_ids, labels = [item.to(device) for item in batch]
            attention_mask = torch.arange(input_ids.size(1)).expand(len(valid_length), input_ids.size(1)).to(device) < valid_length.unsqueeze(1)
            outputs = model(input_ids, attention_mask, token_type_ids)
            predictions = torch.argmax(outputs, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total

# 학습된 모델 로드 및 평가
loaded_model = BERTClassifier(bert_model, dr_rate=0.6).to(device)
loaded_model.load_state_dict(torch.load(model_save_path, map_location=device))
print("Model loaded for evaluation.")

# 평가
train_acc = calc_accuracy(loaded_model, train_dataloader, device)
test_acc = calc_accuracy(loaded_model, test_dataloader, device)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# 감정 예측 함수
def predict_emotion(text, model, tokenizer, device):
    model.eval()  # 평가 모드로 전환
    transform = nlp.data.BERTSentenceTransform(
        tokenizer, max_seq_length=64, pad=True, pair=False
    )
    input_ids, valid_length, token_type_ids = transform([text])
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    valid_length = torch.tensor(valid_length).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(token_type_ids).unsqueeze(0).to(device)
    attention_mask = torch.arange(input_ids.size(1)).expand(len(valid_length), input_ids.size(1)).to(device) < valid_length.unsqueeze(1)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
        predicted_class = torch.argmax(outputs, dim=-1).item()
    label_map = {0: "분노", 1: "행복", 2: "불안", 3: "놀람", 4: "슬픔"}
    return label_map[predicted_class]

# 표준 입력을 강제로 utf-8로 설정
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')

while True:
    try:
        user_input = input("문장을 입력하세요 (종료하려면 'exit' 입력): ")
        if user_input.lower() == 'exit':
            print("테스트를 종료합니다.")
            break
        emotion = predict_emotion(user_input, model, tokenizer, device)
        print(f"입력 문장: {user_input}")
        print(f"예측된 감정: {emotion}")
        print()
    except UnicodeDecodeError as e:
        print(f"문자열 디코딩 오류 발생: {e}")
        continue
