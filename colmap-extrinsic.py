import numpy as np
import struct
import json

def qvec2rotmat(qvec):
    """COLMAP quaternion (w, x, y, z) → 3×3 rotation matrix"""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ], dtype=np.float32)

def colmap_pose_to_extrinsic(qvec, tvec):
    """COLMAP pose → 4×4 Extrinsic matrix"""
    R = qvec2rotmat(qvec)
    extr = np.eye(4, dtype=np.float32)
    extr[:3, :3] = R
    extr[:3, 3] = tvec
    return extr

def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<i", f.read(4))[0]
            qvec = struct.unpack("<dddd", f.read(32))
            tvec = struct.unpack("<ddd", f.read(24))
            camera_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")
            # 2D points skip
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * num_points2D)
            images[name] = {
                "qvec": np.array(qvec, dtype=np.float64),
                "tvec": np.array(tvec, dtype=np.float64),
                "camera_id": camera_id
            }
    return images

def export_extrinsics(images_bin_path, json_path="extrinsics.json", txt_path="extrinsics.txt"):
    images = read_images_binary(images_bin_path)
    output = {}

    with open(txt_path, "w") as txt_file:
        for name, data in images.items():
            extrinsic = colmap_pose_to_extrinsic(data["qvec"], data["tvec"])
            output[name] = extrinsic.tolist()

            txt_file.write(f"Image: {name}\n")
            txt_file.write(np.array2string(extrinsic, formatter={'float_kind':lambda x: f'{x: .6f}'}))
            txt_file.write("\n" + "-" * 40 + "\n")

    # JSON 저장
    with open(json_path, "w") as json_file:
        json.dump(output, json_file, indent=4)

    print(f"[완료] Extrinsics JSON → {json_path}")
    print(f"[완료] Extrinsics TXT → {txt_path}")

# 사용 예시
images_bin_path = "/home/imlab/Downloads/tandt/truck/sparse/0/images.bin"  # ← 실제 경로로 변경
export_extrinsics(images_bin_path, "extrinsics.json", "extrinsics.txt")

