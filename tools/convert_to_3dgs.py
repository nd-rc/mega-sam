import numpy as np
import os
import argparse
import struct
from scipy.spatial.transform import Rotation as R
import cv2
from tqdm import tqdm

def write_cameras_binary(path, width, height, intrinsics):
    """Пишет cameras.bin (PINHOLE model)"""
    with open(path, "wb") as f:
        # NUM_CAMERAS (uint64)
        f.write(struct.pack("<Q", 1))
        
        # CAMERA 1
        cam_id = 1
        model_id = 1 # PINHOLE
        
        # fx, fy, cx, cy
        params = [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]
        
        # ID(int), MODEL(int), W(u64), H(u64)
        f.write(struct.pack("<iiQQ", cam_id, model_id, width, height))
        
        # PARAMS (4 * double)
        for p in params:
            f.write(struct.pack("<d", p))

def write_images_binary(path, images, poses):
    """Пишет images.bin"""
    with open(path, "wb") as f:
        # NUM_IMAGES (uint64)
        f.write(struct.pack("<Q", len(images)))
        
        for i in tqdm(range(len(images)), desc="Exporting images.bin"):
            pose = poses[i]
            
            # Фильтр битых поз
            if np.any(np.isnan(pose)) or np.any(np.isinf(pose)):
                continue

            # Droid (c2w) -> COLMAP (w2c)
            c2w = np.eye(4)
            c2w[:3, :] = pose[:3, :] 
            
            try:
                w2c = np.linalg.inv(c2w)
            except np.linalg.LinAlgError:
                continue

            # Кватернион (scipy: x,y,z,w -> colmap: w,x,y,z)
            q = R.from_matrix(w2c[:3, :3]).as_quat() 
            qw, qx, qy, qz = q[3], q[0], q[1], q[2]
            
            # Трансляция
            tx, ty, tz = w2c[:3, 3]
            
            img_id = i + 1
            cam_id = 1
            img_name = f"{i:05d}.jpg"
            
            # IMAGE_ID(u32), Q(4d), T(3d), CAM_ID(u32)
            f.write(struct.pack("<I4d3dI", img_id, qw, qx, qy, qz, tx, ty, tz, cam_id))
            
            # NAME (string + null byte)
            f.write(img_name.encode("utf-8") + b"\x00")
            
            # NUM_POINTS2D (u64) = 0 (мы не знаем 2D точек)
            f.write(struct.pack("<Q", 0))

def write_points3D_binary(path):
    """Создает заглушку points3D.bin (gsplat требует этот файл)"""
    # Создаем облако случайных точек для инициализации
    num_points = 500
    with open(path, "wb") as f:
        # NUM_POINTS (uint64)
        f.write(struct.pack("<Q", num_points))
        
        for i in range(num_points):
            point_id = i + 1
            xyz = np.random.rand(3) * 4.0 - 2.0
            rgb = np.random.randint(0, 255, 3)
            error = 0.0
            
            # ID(u64), X(d), Y(d), Z(d), R(u8), G(u8), B(u8), ERR(d)
            f.write(struct.pack("<Q3d3Bd", point_id, xyz[0], xyz[1], xyz[2], rgb[0], rgb[1], rgb[2], error))
            
            # TRACK_LENGTH(u64) = 0
            f.write(struct.pack("<Q", 0))

def process_data(npz_path, output_path):
    # Папки
    sparse_path = os.path.join(output_path, "sparse", "0")
    images_path = os.path.join(output_path, "images")
    os.makedirs(sparse_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    # Загрузка
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    images = data["images"]
    poses = data["cam_c2w"] if "cam_c2w" in data else data["poses"]
    intrinsics = data["intrinsic"]
    H, W, _ = images[0].shape

    print(f"Processing {len(images)} frames...")

    # 1. Сохраняем картинки (JPG)
    for i in tqdm(range(len(images)), desc="Saving JPGs"):
        save_img_path = os.path.join(images_path, f"{i:05d}.jpg")
        if not os.path.exists(save_img_path):
            cv2.imwrite(save_img_path, cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))

    # 2. Удаляем старые .txt файлы (чтобы не мешали)
    for f in ["cameras.txt", "images.txt", "points3D.txt"]:
        p = os.path.join(sparse_path, f)
        if os.path.exists(p):
            os.remove(p)

    # 3. Пишем бинарники
    print("Writing binary COLMAP files...")
    write_cameras_binary(os.path.join(sparse_path, "cameras.bin"), W, H, intrinsics)
    write_images_binary(os.path.join(sparse_path, "images.bin"), images, poses)
    write_points3D_binary(os.path.join(sparse_path, "points3D.bin"))

    print("✅ Conversion DONE. Binary files created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    process_data(args.npz_path, args.output_path)