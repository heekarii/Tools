import os
import cv2

in_dir = './crops/crop_00000.png'
out_dir = './res'

tar_h, tar_w = 500, 500

def upscale_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(input_dir)
    if img is None:
        print(f"[경고] 이미지 로딩 실패: {input}")
        return
    h, w = img.shape[:2]
    resimg = cv2.resize(img, (tar_w, tar_h), interpolation=cv2.INTER_CUBIC)
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_dir))[0]}_res.png")
    cv2.imwrite(output_path, resimg)
    print(f"저장 완료: {output_path}")

def main():
    upscale_images(in_dir, out_dir)

if __name__ == "__main__":
    main()
