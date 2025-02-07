import torch
from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline

# 모델과 토크나이저 로드
model = BertForSequenceClassification.from_pretrained('./trained_model')
tokenizer = BertTokenizer.from_pretrained('klue/bert-base')

# 파이프라인 설정
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework='pt')

# 새로운 데이터에 대한 테스트
test_texts = [
    "최근에 일이 너무 힘들어서 스트레스를 많이 받고 있어요.",
    "정말 잘하고 계세요, 항상 최선을 다하는 모습이 보여요."
]
for text in test_texts:
    result = pipeline(text)
    print(f"Input: {text}\nPredicted label: {result[0]['label']}")

