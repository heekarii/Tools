import os
import argparse
import lpips
import torch
import torchvision.transforms as transforms
from PIL import Image
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="LPIPS 비교 후 CSV 저장")
    parser.add_argument('--target', required=True, help='타겟 이미지 폴더 경로')
    parser.add_argument('--gt', required=True, help='GT 이미지 폴더 경로')
    parser.add_argument('--output', required=True, help='결과 CSV 경로')
    return parser.parse_args()

def load_and_preprocess(img_path, size):
    img = Image.open(img_path).convert('RGB').resize(size, Image.BICUBIC)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(img).unsqueeze(0)

def main():
    args = parse_args()
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.eval()
    sum = 0
    results = []
    for filename in sorted(os.listdir(args.gt)):  # ✅ GT 기준으로 반복
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        base = os.path.splitext(filename)[0]
        sr_name = f"{base}_rlt.png"

        gt_path = os.path.join(args.gt, filename)
        target_path = os.path.join(args.target, sr_name)

        if not os.path.exists(target_path):
            print(f"[경고] 타겟 없음: {sr_name}")
            continue


        gt_img_pil = Image.open(gt_path).convert('RGB')
        gt_size = gt_img_pil.size  # (width, height)

        gt_tensor = load_and_preprocess(gt_path, gt_size)
        target_tensor = load_and_preprocess(target_path, gt_size)

        with torch.no_grad():
            dist = loss_fn(target_tensor, gt_tensor)

        lpips_value = dist.item()
        results.append((filename, lpips_value))
        print(f"{filename}: LPIPS = {lpips_value:.4f}")
        sum += lpips_value
    print(f"\nAVERAGE LPIPS: {sum / len(results):.4f}") 

    with open(args.output, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "lpips"])
        writer.writerows(results)
        writer.writerow(["AVERAGE", f"{sum / len(results):.4f}"])

    print(f"\n✅ LPIPS 결과 저장 완료: {args.output}")

if __name__ == "__main__":
    main()
