import os
import random
import glob
import json

# 이미지 및 JSON 파일 경로
IMG_DIR = "img"
DOC_DIR = "doc"

def get_random_car_data(num_samples=3):
    # img 폴더에서 모든 이미지 파일(.jpg, .png 등) 가져오기
    img_files = glob.glob(os.path.join(IMG_DIR, "*.*"))

    # 확장자 제거 후 파일 이름만 추출
    img_names = [os.path.splitext(os.path.basename(f))[0] for f in img_files]

    # 파일이 많으므로 랜덤하게 3개 선택
    selected_names = random.sample(img_names, num_samples)

    # 선택된 이미지와 대응하는 JSON 파일 가져오기
    selected_data = [
        {
            "image": os.path.join(IMG_DIR, f"{name}.jpg"),  # 확장자는 jpg로 가정 (필요 시 변경)
            "description": os.path.join(DOC_DIR, f"{name}.json")
        }
        for name in selected_names
    ]

    return selected_data

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 테스트 실행
if __name__ == "__main__":
    car_data = get_random_car_data()
    for data in car_data:
        print(f"이미지: {data['image']}, 설명 파일: {data['description']}")
    
    # JSON 파일 예시 로딩
    if car_data:
        json_path = car_data[0]['description']  # 첫 번째 JSON 파일 로드
        car_info = load_json(json_path)
        print(f"차량 이름: {car_info['name']}")
        print(f"제조사: {car_info['manufacturer']}")
        print(f"출시 연도: {car_info['year']}")
        print(f"엔진: {car_info['engine']}")
        print(f"마력: {car_info['horsepower']} HP")
        print(f"연비: {car_info['fuel_efficiency']}")
        print(f"설명: {car_info['description']}")
