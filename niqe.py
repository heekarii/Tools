import os
import cv2
import numpy as np
import pandas as pd
from torchvision.transforms.functional import rgb_to_grayscale
from basicsr.metrics.niqe import calculate_niqe

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))

def calculate_niqe_for_folder(folder_path, output_csv='niqe_scores.csv',
                              use_y_channel=True, crop_border=0):
    """
    use_y_channel=True  -> Y(밝기) 채널 기준 NIQE (권장)
    use_y_channel=False -> RGB 전체 기준 NIQE
    crop_border: 가장자리 픽셀 크기만큼 잘라내고 평가 (필요 없으면 0)
    """
    results = []
    sum = 0
    for filename in sorted(os.listdir(folder_path)):
        if not is_image_file(filename):
            continue

        file_path = os.path.join(folder_path, filename)
        img_bgr = cv2.imread(file_path)

        if img_bgr is None:
            print(f"[경고] 이미지를 열 수 없음: {filename}")
            continue

        # OpenCV는 BGR이므로 RGB로 변환
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            score = calculate_niqe(
                img_rgb,
                crop_border=crop_border,
                use_padding=True,          # 작은 이미지에서도 안정적으로 동작하도록 패딩 사용
                convert_to='y' if use_y_channel else 'rgb'
            )
            results.append({'filename': filename, 'niqe_score': float(score)})
            sum += float(score)
            print(f"{filename}: NIQE = {float(score):.4f}")
        except Exception as e:
            print(f"[오류] {filename} 처리 중 오류 발생: {e}")
    print(f"\nAVERAGE NIQE: {sum / len(results):.4f}")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nCSV 저장 완료: {output_csv}")

# 사용 예시
folder_path = './orchids_bicubic'  # NIQE를 계산할 폴더 경로
output_csv = './orchids_bicubic_niqe.csv'
calculate_niqe_for_folder(folder_path, output_csv, use_y_channel=True, crop_border=0)
