import numpy as np
import os
import argparse
import shutil
from scipy.spatial.transform import Rotation as R
import cv2
from tqdm import tqdm

def save_colmap(path, images, poses, intrinsics):
    os.makedirs(os.path.join(path, "sparse/0"), exist_ok=True)
    os.makedirs(os.path.join(path, "images"), exist_ok=True)

    # 1. cameras.txt
    # Model: PINHOLE
    # Parameters: fx, fy, cx, cy
    H, W, C = images[0].shape
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    
    with open(os.path.join(path, "sparse/0/cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")

    # 2. images.txt
    with open(os.path.join(path, "sparse/0/images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for i in tqdm(range(len(images)), desc="Exporting images"):
            # Droid saves Camera-to-World (c2w). COLMAP needs World-to-Camera (w2c).
            c2w = np.eye(4)
            c2w[:3, :] = poses[i][:3, :]
            w2c = np.linalg.inv(c2w)
            
            q = R.from_matrix(w2c[:3, :3]).as_quat() # x, y, z, w
            t = w2c[:3, 3]
            
            # COLMAP expects qw, qx, qy, qz
            img_name = f"{i:05d}.jpg"
            f.write(f"{i+1} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} 1 {img_name}\n")
            f.write("\n") # Empty points line
            
            # Save image physically
            save_img_path = os.path.join(path, "images", img_name)
            cv2.imwrite(save_img_path, cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))

    # 3. points3D.txt (Create empty, 3DGS will generate its own or use sparse later)
    with open(os.path.join(path, "sparse/0/points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", required=True, help="Path to droid output .npz")
    parser.add_argument("--output_path", required=True, help="Where to save COLMAP format")
    args = parser.parse_args()

    print(f"Loading {args.npz_path}...")
    data = np.load(args.npz_path)
    
    # Extract data
    images = data["images"]
    # Droid saves poses as camera-to-world matrices (N, 4, 4) or (N, 3, 4)
    # The output from test_demo.py hack we used usually gives cam_c2w as (N, 4, 4)
    poses = data["cam_c2w"] 
    intrinsics = data["intrinsic"] # This is usually just 3x3 K matrix

    print(f"Found {len(images)} frames.")
    save_colmap(args.output_path, images, poses, intrinsics)
    print("Done! Ready for Gaussian Splatting.")


#    python tools/convert_to_3dgs.py \
#  --npz_path "../input/my_apartment/droid.npz" \
#  --output_path "./input/my_apartment/gaussian_input"