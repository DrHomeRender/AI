import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd


class ChatDataset(Dataset):
    def __init__(self, filename):
        data = pd.read_csv(filename)
        self.texts = data['text'].tolist()
        self.labels = data['label'].tolist()
        self.tokenizer = BertTokenizer.from_pretrained('klue/bert-base')

        # 문자열 라벨을 숫자로 변환
        self.label_mapping = {'support': 0, 'encouragement': 1}
        self.labels = [self.label_mapping[label] for label in self.labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokenized_input = self.tokenizer(self.texts[idx], return_tensors='pt', padding='max_length', truncation=True,
                                         max_length=512)
        input_ids = tokenized_input['input_ids'].squeeze()
        attention_mask = tokenized_input['attention_mask'].squeeze()

        # 정수형 라벨 변환
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}


# 토크나이저와 모델 로드
tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
model = BertForSequenceClassification.from_pretrained('klue/bert-base', num_labels=2)  # support와 encouragement 두 개의 클래스

# 데이터셋 생성
train_dataset = ChatDataset('data.csv')

# 트레이닝 아규먼트 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=100,
    per_device_train_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    save_strategy="epoch"
)

# 트레이너 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# 트레이닝 시작
trainer.train()

# 모델 저장
model.save_pretrained('./trained_model')
