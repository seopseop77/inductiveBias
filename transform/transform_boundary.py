import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import *

def extract_boundary(image_path):
    # 1. 이미지 로드 (그레이스케일)
    image_path = str(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 이미지가 112x112가 아닐 경우를 대비해 크기 확인 (선택 사항)
    # img = cv2.resize(img, (112, 112))

    # 2. 이진화 확인 (이미 binary라면 생략 가능하지만 안전을 위해 수행)
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 3. 커널 생성 (3x3 크기)
    kernel = np.ones((3, 3), np.uint8)

    # 4. 침식 연산 (Erosion): 도형을 1픽셀씩 깎아냅니다.
    eroded_img = cv2.erode(binary_img, kernel, iterations=1)

    # 5. 경계 추출: 원본 - 침식 이미지
    boundary = cv2.subtract(binary_img, eroded_img) 

    # 6. 경계 이미지 저장
    name = "boundary"
    boundary_path = TRANSFORM_DATA_DIR / (image_path.split("/")[-1].split(".")[0] + "_" + name + ".png")
    cv2.imwrite(boundary_path, boundary)

    return boundary_path 

if __name__ == "__main__":
    # 실행 및 시각화
    image_path = "dataset/images/1.png"  # 파일 경로를 입력하세요
    boundary_path = extract_boundary(image_path)

    # 결과 확인
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Binary Image")
    plt.imshow(cv2.imread(image_path), cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Extracted Boundary")
    plt.imshow(cv2.imread(boundary_path), cmap='gray')

    plt.show()

    # 필요 시 이미지 저장
    # cv2.imwrite('boundary_result.png', edge)