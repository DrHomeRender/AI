# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 21:29:54 2025

@author: DrHomeRander
"""

import os
import random
import glob
import json
import cv2  # OpenCV 라이브러리 추가

IMG_DIR = "img"
DOC_DIR = "doc"

def select_data(num_samples=3):
    # img 폴더에서 모든 이미지 파일 가져옴
    img_list = glob.glob(os.path.join(IMG_DIR, "*.jpg"))  # 확장자가 jpg인 파일만 가져옴
    img_names = [os.path.splitext(os.path.basename(f))[0] for f in img_list]

    selected_names = random.sample(img_names, num_samples)

    selected_data = [
        {
            "image": os.path.join(IMG_DIR, f"{name}.jpg"),
            "description": os.path.join(DOC_DIR, f"{name}.json")
        }
        for name in selected_names
    ]
    
    return selected_data

def load_and_display_data(car_data):
    for data in car_data:
        # 이미지 파일 로드
        image = cv2.imread(data['image'])
        if image is not None:
            cv2.imshow('Car Image', image)  # 이미지 출력
            cv2.waitKey(0)  # 키 입력 대기 (아무 키나 누르면 다음 이미지로 넘어감)
            cv2.destroyAllWindows()  # 열린 모든 창 닫기

        # 설명 파일 로드 및 출력
        with open(data['description'], 'r', encoding='utf-8') as file:
            description = json.load(file)
            print(f"설명: {description['description']}")

if __name__ == "__main__":
    car_data = select_data()  # 함수 이름을 'select_data'로 수정
    load_and_display_data(car_data)  # JSON 파일을 읽고 이미지를 출력하는 추가 함수
