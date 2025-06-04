import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import csv

def compare_resized_sr_to_gt(gt_dir, sr_dir, log_path="psnr_ssim_log.csv"):
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    psnr_list, ssim_list, log_rows = [], [], []

    for fname in gt_files:
        base = os.path.splitext(fname)[0]
        sr_name = base + '_x4_SR.png'
        gt_path = os.path.join(gt_dir, fname)
        sr_path = os.path.join(sr_dir, sr_name)

        if not os.path.exists(sr_path):
            print(f"❌ {sr_name} SR 파일 없음, 건너뜀")
            continue

        gt = np.array(Image.open(gt_path).convert('RGB')).astype(np.float32)
        sr = Image.open(sr_path).convert('RGB').resize(gt.shape[1::-1], Image.BICUBIC)
        sr = np.array(sr).astype(np.float32)

        if gt.shape != sr.shape:
            print(f"⚠️ {fname} 리사이즈 후에도 크기 다름? → {gt.shape} vs {sr.shape}")
            continue

        psnr_val = psnr(gt, sr, data_range=255)
        ssim_val = ssim(gt, sr, data_range=255, channel_axis=2)

        print(f"{fname} ▶️ PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        log_rows.append([fname, f"{psnr_val:.2f}", f"{ssim_val:.4f}"])

    if psnr_list:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        print(f"\n✅ 평균 PSNR: {avg_psnr:.2f} dB")
        print(f"✅ 평균 SSIM: {avg_ssim:.4f}")
        log_rows.append(["AVERAGE", f"{avg_psnr:.2f}", f"{avg_ssim:.4f}"])
    else:
        print("\n⚠️ 비교 가능한 이미지가 없음.")

    # CSV 로그 저장
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "PSNR (dB)", "SSIM"])
        writer.writerows(log_rows)

    print(f"\n📁 로그 저장됨: {log_path}")

# 예시 실행 (경로 맞게 수정!)
compare_resized_sr_to_gt(
    gt_dir="./GT",
    sr_dir= "./tar",
    log_path="GT-EDSR_x4.csv"
)
