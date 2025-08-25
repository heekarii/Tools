import os
import torch
import pyiqa
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 화질 평가 모델 로드
niqe_metric = pyiqa.create_metric('niqe', device=device)
musiq_metric = pyiqa.create_metric('musiq', device=device)
fid_metric = pyiqa.create_metric('fid', device=device)

# 이미지 로드용 transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

def load_tensor_image(path):
    img = Image.open(path).convert('RGB')
    img = transform(img)
    return img.unsqueeze(0).to(device)

def evaluate_folder(folder_path, output_csv='niqe_musiq.csv'):
    results = []
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for fname in tqdm(files):
        try:
            img_path = os.path.join(folder_path, fname)
            img_tensor = load_tensor_image(img_path)
            row = {'filename': fname}

            try:
                row['NIQE'] = niqe_metric(img_tensor).item()
            except Exception as e:
                print(f"[NIQE 오류] {fname}: {e}")
                row['NIQE'] = None

            try:
                row['MUSIQ'] = musiq_metric(img_tensor).item()
            except Exception as e:
                print(f"[MUSIQ 오류] {fname}: {e}")
                row['MUSIQ'] = None

            results.append(row)

        except Exception as e:
            print(f"[로드 오류] {fname}: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\nCSV 저장 완료: {output_csv}")
    else:
        print("[경고] 유효한 결과가 없습니다. CSV 생략됨.")

def evaluate_fid(gt_folder, sr_folder, csv_path):
    try:
        fid_score = fid_metric(gt_folder, sr_folder).item()
        print(f"\n[FID 전체 점수]: {fid_score:.4f}")

        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            df = pd.read_csv(csv_path)
            df['FID_global'] = fid_score
            df.to_csv(csv_path, index=False)
            print(f"FID 포함된 CSV 저장 완료: {csv_path}")
        else:
            print("CSV 파일이 비어 있어 FID 병합 생략됨.")

    except Exception as e:
        print(f"[FID 오류]: {e}")

# 경로 설정
gt_folder = './horns_gt'
sr_folder = './horns_esrgan'
output_csv = 'horns_esrgan_quality.csv'

# 실행
evaluate_folder(sr_folder, output_csv)
evaluate_fid(gt_folder, sr_folder, output_csv)
