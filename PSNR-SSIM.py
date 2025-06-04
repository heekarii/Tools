import os
import csv
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_psnr_ssim(hr_dir, sr_dir, log_path="psnr_ssim_log.csv"):
    hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    psnr_list, ssim_list = [], []
    log_rows = []

    for fname in hr_files:

        hr_path = os.path.join(hr_dir, fname)
        fname_core = os.path.splitext(fname)[0]
        sr_fname = fname_core + "_x4_SR.png"
        sr_path = os.path.join(sr_dir, sr_fname)
        #sr_path = os.path.join(sr_dir, fname)

        if not os.path.exists(sr_path):
            print(f"❌ {sr_fname}는 SR 폴더에 없음. 건너뜀.")
            continue

        hr = np.array(Image.open(hr_path).convert('RGB')).astype(np.float32)
        sr = np.array(Image.open(sr_path).convert('RGB')).astype(np.float32)

        if hr.shape != sr.shape:
            print(f"⚠️ {fname} 크기 다름: {hr.shape} vs {sr.shape} → 건너뜀.")
            continue

        psnr_val = psnr(hr, sr, data_range=255)
        ssim_val = ssim(hr, sr, data_range=255, channel_axis=2)

        print(f"{fname} ▶️ PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        log_rows.append([fname, f"{psnr_val:.2f}", f"{ssim_val:.4f}"])

    # 평균 계산 및 출력
    if psnr_list:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        print(f"\n✅ 평균 PSNR: {avg_psnr:.2f} dB")
        print(f"✅ 평균 SSIM: {avg_ssim:.4f}")
        log_rows.append(["AVERAGE", f"{avg_psnr:.2f}", f"{avg_ssim:.4f}"])
    else:
        print("\n⚠️ 비교할 수 있는 이미지가 없음.")

    # CSV 파일로 저장
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "PSNR (dB)", "SSIM"])
        writer.writerows(log_rows)

    print(f"\n📁 로그 저장 완료: {log_path}")

# ✅ 경로 지정 후 실행
calculate_psnr_ssim(hr_dir="./GT", sr_dir="./tar", log_path="psnr_ssim_log.csv")
