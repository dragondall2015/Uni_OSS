# 필요한 패키지 설치
# !pip install mxnet
# !pip install gluonnlp pandas tqdm
# !pip install sentencepiece
# !pip install transformers==4.8.1
# !pip install torch==1.10.1
# !pip install git+https://git@github.com/SKTBrain/KoBERT.git@master

# cpu 에서 1000개 데이터로 테스트
# 패키지 임포트
import torch
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
chatbot_data = pd.read_csv('../../data_set/all_emotions_oversampled.csv')

# 두번째 데이터셋 위치
# chatbot_data = pd.read_csv('../data_set/emotion_merge_subemotion_text.csv')

# 감정 레이블 매핑
# 첫번째 데이터 셋 레이블 all_emotions_combined.csv
label_dict = {"분노": 0, "행복": 1, "불안": 2, "놀람": 3, "슬픔": 4}

# 두번째 데이터 셋 레이블 emotion_merge_subemotion_text.csv
# 상처 슬픔을 하나로 묶음
# label_dict = {"기쁨": 0, "당황": 1, "분노": 2, "불안": 3, "상처": 4, "슬픔": 4}
# chatbot_data['emotion'] = chatbot_data['emotion'].map(label_dict)
# data_list = chatbot_data[['text', 'emotion']].values.tolist()

# 샘플 데이터 준비 (1000개 샘플)
sample_data = data_list[:1000]
train_data, test_data = train_test_split(sample_data, test_size=0.25, random_state=42)

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
batch_size = 16
max_len = 64

train_dataset = BERTDataset(train_data, tokenizer, max_len)
test_dataset = BERTDataset(test_data, tokenizer, max_len)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 정의
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=5, dr_rate=0.5):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(p=dr_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # KoBERT의 출력에서 첫 번째 요소(embedding layer) 사용
        sequence_output, pooler_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        cls_token = sequence_output[:, 0]  # [CLS] 토큰
        pooled_output = self.dropout(cls_token)
        return self.classifier(pooled_output)

# CPU 모드 설정
device = torch.device("cpu")
model = BERTClassifier(bert_model, dr_rate=0.5).to(device)

# 학습 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 학습 루프
num_epochs = 3
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

# 평가
train_acc = calc_accuracy(model, train_dataloader, device)
test_acc = calc_accuracy(model, test_dataloader, device)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")