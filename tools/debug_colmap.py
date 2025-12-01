import pycolmap
import os
import sys

# Путь к данным
data_dir = "/mnt/d/work/data/room/gaussian_input"
sparse_path = os.path.join(data_dir, "sparse", "0")
images_path = os.path.join(data_dir, "images")

print(f"Checking path: {sparse_path}")

if not os.path.exists(sparse_path):
    print("❌ Error: Sparse folder does not exist!")
    sys.exit(1)

try:
    # Пытаемся загрузить реконструкцию напрямую
    manager = pycolmap.SceneManager(sparse_path)
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()
    
    print(f"✅ Cameras found: {len(manager.cameras)}")
    print(f"✅ Images found: {len(manager.images)}")
    print(f"✅ Points found: {len(manager.points3D)}")
    
    if len(manager.images) == 0:
        print("❌ Warning: Images list is empty. pycolmap failed to parse images.txt or find image files.")
        print(f"Checking if image folder exists at: {images_path}")
        print(f"Exists: {os.path.exists(images_path)}")
        if os.path.exists(images_path):
             print(f"File count: {len(os.listdir(images_path))}")
             print("First file:", os.listdir(images_path)[0])

except Exception as e:
    print(f"❌ CRITICAL ERROR loading COLMAP: {e}")