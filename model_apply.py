# 패키지 임포트
import torch
import torch.nn.functional as F
import sys
import io
import os
import gluonnlp as nlp
from torch import nn
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

# KoBERT 모델 및 Vocabulary 로드
bert_model, vocab = get_pytorch_kobert_model()
tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)

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

# GPU 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_per_process_memory_fraction(0.3, device)

# 모델 초기화 및 로드
model_save_path = "model/hyper2_kobert_emotion_model.pth"
model = BERTClassifier(bert_model, dr_rate=0.6).to(device)

if not torch.cuda.is_available():
    print("CUDA가 사용 불가능합니다. CPU 모드로 실행합니다.")

if torch.cuda.is_available() and not torch.cuda.is_initialized():
    print("CUDA 초기화 중...")

if os.path.exists(model_save_path):
    print("학습된 모델을 로드합니다...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
else:
    raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_save_path}")

# 감정 예측 함수
def predict_emotion_with_probabilities(text, model, tokenizer, device):
    model.eval()
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
        probabilities = F.softmax(outputs, dim=-1)

    probabilities = probabilities.squeeze().cpu().numpy()
    # 숫자 인덱스에 따른 감정 이름 매핑
    label_map = {0: "분노", 1: "행복", 2: "불안", 3: "놀람", 4: "슬픔"}
    emotion_probabilities = {label_map[i]: float(probabilities[i]) for i in range(len(probabilities))}
    predicted_emotion = max(emotion_probabilities, key=emotion_probabilities.get)
    return predicted_emotion, emotion_probabilities

