import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# ✅ 1. KoGPT2 모델 및 토크나이저 로드
model_name = "skt/kogpt2-base-v2"
tokenizer = GPT2TokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
)

# ✅ 2. Padding Token 설정 (GPT2는 기본적으로 padding token이 없음)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token  # 또는 '[PAD]' 사용 가능

# ✅ 3. 모델 로드 및 Token Embeddings 조정
model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))  # 추가된 PAD 토큰 반영

# ✅ 4. 데이터셋 로드
dataset = load_dataset("text", data_files={"train": "gpt2_train_data.txt"})

# 🔍 디버깅: 데이터셋 확인
print("\n🔹 원본 데이터셋 샘플 확인:")
print(dataset["train"][0])  # 첫 번째 데이터 확인

# ✅ 5. 데이터 토크나이징 함수 (배치 처리)
def tokenize_function(examples):
    texts = examples["text"]  # 🔹 리스트로 들어온 문장을 개별적으로 처리

    # 🔍 디버깅: 입력 데이터 확인
    print("\n🔹 tokenize_function 입력 예제:")
    print(texts)

    # ✅ 개별 문장에 대해 Tokenize 수행
    tokenized_texts = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # 🔍 디버깅: 토큰화된 결과 확인
    print("\n🔹 토큰화된 데이터 샘플:")
    print(tokenized_texts)

    return tokenized_texts

# ✅ 6. 데이터셋 변환 (토크나이징 적용)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 🔍 디버깅: 변환된 데이터셋 확인
print("\n🔹 변환된 데이터셋 샘플:")
print(tokenized_datasets["train"][0])

# ✅ 7. 데이터 Collator 설정 (Padding 적용)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ✅ 8. 훈련 설정
training_args = TrainingArguments(
    output_dir="./kogpt2_results",
    num_train_epochs=5,  # 학습 Epochs
    per_device_train_batch_size=4,  # 배치 크기
    per_device_eval_batch_size=4,
    save_strategy="epoch",  # 매 Epoch 마다 저장
    evaluation_strategy="no",  # 평가 수행 안 함 (오류 해결)
    logging_dir="./kogpt2_logs",
    logging_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    save_total_limit=2,
    load_best_model_at_end=False,  # 🔹 평가 없이 가장 좋은 모델 저장 안 함 (오류 해결)
    report_to="none"
)
# ✅ 9. Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator
)

# ✅ 10. 훈련 시작
trainer.train()

# ✅ 11. 훈련된 모델 저장
model.save_pretrained("./trained_kogpt2")
tokenizer.save_pretrained("./trained_kogpt2")
print("✅ KoGPT2 학습 완료 및 저장됨!")
