from transformers import AutoModelForCausalLM, AutoTokenizer

# KoGPT 모델 및 토크나이저 불러오기
MODEL_NAME = "skt/kogpt2-base-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# 사용자 질문 입력
user_input = "오늘의 자동차 3가지를 소개해줘."

# 모델이 이해할 수 있도록 변환 (토큰화)
input_ids = tokenizer.encode(user_input, return_tensors="pt")

# 모델이 답변 생성
output_ids = model.generate(input_ids, max_length=50, do_sample=True, top_k=50)

# 생성된 답변 디코딩
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("HELPER:", response)
