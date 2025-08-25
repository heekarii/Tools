import os
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="폴더 내 이미지 Bicubic 4배 업스케일링")
    parser.add_argument('--input', required=True, help='입력 이미지 폴더 경로')
    parser.add_argument('--output', required=True, help='출력 폴더 경로')
    return parser.parse_args()

def upscale_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_x4.png")

        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[경고] 이미지 로딩 실패: {filename}")
            continue

        h, w = img.shape[:2]
        upscaled = cv2.resize(img, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(output_path, upscaled)
        print(f"{filename} → 저장 완료: {output_path}")

def main():
    args = parse_args()
    upscale_images(args.input, args.output)

if __name__ == "__main__":
    main()
