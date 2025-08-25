import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from scipy.linalg import sqrtm
from torch.nn.functional import adaptive_avg_pool2d
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_image_files(folder):
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def load_and_preprocess_image(path):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # -1 ~ 1
    ])
    return transform(img).unsqueeze(0)  # (1, 3, 299, 299)

def extract_features(paths, model, batch_size=16):
    features = []

    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch_size), desc="Extracting features"):
            batch_paths = paths[i:i+batch_size]
            batch_images = [load_and_preprocess_image(p) for p in batch_paths]
            batch_tensor = torch.cat(batch_images, dim=0).to(device)

            pred = model(batch_tensor)[0]  # pool3
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1)).squeeze(-1).squeeze(-1)  # (B, 2048)
            features.append(pred.cpu().numpy())

    return np.concatenate(features, axis=0)

def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def compute_fid_score(folder1, folder2):
    paths1 = get_image_files(folder1)
    paths2 = get_image_files(folder2)

    model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    # Inception에서 마지막 fc 레이어 제거하고 feature만 추출
    model.fc = torch.nn.Identity()

    feats1 = extract_features(paths1, model)
    feats2 = extract_features(paths2, model)

    mu1, sigma1 = feats1.mean(axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = feats2.mean(axis=0), np.cov(feats2, rowvar=False)

    fid_value = calculate_fid(mu1, sigma1, mu2, sigma2)
    print(f"\nFID score: {fid_value:.4f}")

# 사용 예시
folder_real = './horns_gt'       # GT 이미지 폴더
folder_fake = './horns_edsr'       # 생성 이미지 폴더
compute_fid_score(folder_real, folder_fake)
