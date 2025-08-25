import cv2
import os
import matplotlib.pyplot as plt

# 크롭할 좌표 설정 (x, y, width, height)
crop_box = (350//4, 2250//4, 500//4, 500//4)  # 예시: x=180, y=290, w=100, h=100

# 비교할 이미지들
image_dir = './lll'  # 폴더에 이미지 모아두기
output_dir = './crops'
os.makedirs(output_dir, exist_ok=True)

for filename in sorted(os.listdir(image_dir)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        x, y, w, h = crop_box

        if (filename=='render.png'):
            annotated = img.copy()
            cv2.rectangle(annotated, (int(x - w // 2), int(y -  h  // 2)), (int(x + w // 2), int(y + h // 2)), (0, 0, 255), 2)
            annotated_path = os.path.join(output_dir, f"annotated_{filename}")
            cv2.imwrite(annotated_path, annotated)
            #print(f"Saved: {save_path}")

            crop = img[y - h // 2 :y + h // 2, x- w // 2 : x + w // 2]
            save_path = os.path.join(output_dir, f"crop_{filename}")
            cv2.imwrite(save_path, crop)

            continue
        
        crop = img[y - h//2:y+h//2, x- w //2 : x + w //2]

        save_path = os.path.join(output_dir, f"crop_{filename}")
        cv2.imwrite(save_path, crop)


        print(f"Saved: {save_path}")
