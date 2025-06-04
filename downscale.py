import cv2
import os
import argparse

def is_image_file(filename):
    # 지원되는 이미지 확장자 목록
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))

def downsample_folder(folder_path, scale):
    # 폴더 내 이미지 파일들에 대해 반복
    for filename in os.listdir(folder_path):
        if is_image_file(filename):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"이미지를 불러올 수 없습니다: {filename}")
                continue

            h, w = img.shape[:2]
            new_w = int(w / scale)
            new_h = int(h / scale)

            # Bicubic downsample
            downsampled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            # 저장 경로 구성
            base, ext = os.path.splitext(filename)
            save_name = f"{base}_x{scale:.1f}{ext}"
            save_path = os.path.join(folder_path, save_name)

            cv2.imwrite(save_path, downsampled)
            print(f"저장 완료: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="이미지들이 있는 폴더 경로")
    parser.add_argument("--scale", type=float, required=True, help="다운샘플 배율 (예: 2.0)")

    args = parser.parse_args()

    if not os.path.isdir(args.image):
        raise NotADirectoryError(f"유효한 폴더가 아닙니다: {args.image}")

    downsample_folder(args.image, args.scale)
