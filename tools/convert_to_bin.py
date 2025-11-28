import os
import struct
import numpy as np

# --- НАСТРОЙКИ ---
BASE_DIR = "/mnt/d/work/input/my_apartment/gaussian_input"
SPARSE_TXT_DIR = os.path.join(BASE_DIR, "sparse/0")
OUTPUT_DIR = SPARSE_TXT_DIR 
# -----------------

def read_cameras_text(path):
    cameras = {}
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return cameras
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(x) for x in parts[4:]])
            cameras[camera_id] = (model, width, height, params)
    return cameras

def read_images_text(path):
    images = {}
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return images
    
    with open(path, "r") as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        
        parts = line.split()
        # Ожидаем минимум 9 полей + имя файла
        if len(parts) < 10:
            i += 1
            continue

        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]
        
        images[image_id] = (qw, qx, qy, qz, tx, ty, tz, camera_id, name)
        
        # Пропускаем следующую строку (точки 2D), даже если она пустая
        i += 2 
    return images

def write_cameras_binary(cameras, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam_id, (model, w, h, params) in cameras.items():
            # gsplat/colmap используют PINHOLE=1. 
            # Внимание: ID модели должен быть int (4 байта)
            model_id = 1 
            f.write(struct.pack("<iiQQ", cam_id, model_id, w, h))
            for p in params:
                f.write(struct.pack("<d", p))

def write_images_binary(images, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img_id, (qw, qx, qy, qz, tx, ty, tz, cam_id, name) in images.items():
            # IMAGE_ID (uint32), Q(4d), T(3d), CAM_ID (uint32)
            f.write(struct.pack("<I4d3dI", img_id, qw, qx, qy, qz, tx, ty, tz, cam_id))
            
            name_bytes = name.encode("utf-8") + b"\x00"
            f.write(name_bytes)
            
            # POINTS2D count (uint64) = 0
            f.write(struct.pack("<Q", 0))

def ensure_points3D_binary(path):
    # Генерируем заглушку, так как DroidSLAM точки не переносим
    num_points = 500
    print(f"Generating dummy points3D.bin with {num_points} random points...")
    
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", num_points))
        for i in range(num_points):
            point_id = i + 1
            xyz = np.random.rand(3) * 4.0 - 2.0
            rgb = np.random.randint(0, 255, 3)
            error = 0.0
            
            # ИСПРАВЛЕНИЕ: error должен быть double ('d'), а не float ('f')
            # Format: ID(u64), X(d), Y(d), Z(d), R(u8), G(u8), B(u8), ERR(d)
            f.write(struct.pack("<Q3d3Bd", point_id, xyz[0], xyz[1], xyz[2], rgb[0], rgb[1], rgb[2], error))
            
            # TRACK_LENGTH(u64) = 0
            f.write(struct.pack("<Q", 0))

def main():
    print(f"Fixing binary files in {OUTPUT_DIR}...")
    
    # Сначала удалим старые бинарники, чтобы не было конфликтов
    for fname in ["cameras.bin", "images.bin", "points3D.bin"]:
        p = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(p):
            os.remove(p)

    # Читаем ТЕКСТОВЫЕ файлы (убедись, что они лежат в sparse/0 или text_backup)
    # Если ты переместил их в text_backup, поправь путь ниже:
    txt_source = SPARSE_TXT_DIR
    if not os.path.exists(os.path.join(txt_source, "images.txt")):
        txt_source = os.path.join(SPARSE_TXT_DIR, "text_backup")
        print(f"Looking for text files in {txt_source}...")

    cams = read_cameras_text(os.path.join(txt_source, "cameras.txt"))
    imgs = read_images_text(os.path.join(txt_source, "images.txt"))

    if not cams or not imgs:
        print("CRITICAL ERROR: Could not read source text files!")
        return

    write_cameras_binary(cams, os.path.join(OUTPUT_DIR, "cameras.bin"))
    write_images_binary(imgs, os.path.join(OUTPUT_DIR, "images.bin"))
    ensure_points3D_binary(os.path.join(OUTPUT_DIR, "points3D.bin"))
    
    print("DONE. Try running simple_trainer again.")

if __name__ == "__main__":
    main()