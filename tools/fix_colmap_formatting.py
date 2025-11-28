import os

# Путь к папке sparse/0
SPARSE_DIR = "/mnt/d/work/input/my_apartment/gaussian_input/sparse/0"
CAMERAS_FILE = os.path.join(SPARSE_DIR, "cameras.txt")
IMAGES_FILE = os.path.join(SPARSE_DIR, "images.txt")

def fix_colmap():
    print(f"Processing files in {SPARSE_DIR}...")

    # --- ШАГ 1: Исправляем cameras.txt ---
    if not os.path.exists(CAMERAS_FILE):
        print("Error: cameras.txt not found!")
        return

    with open(CAMERAS_FILE, 'r', encoding='utf-8') as f:
        cam_lines = f.readlines()
    
    new_cam_lines = []
    camera_model = "PINHOLE" # Дефолт, если не удастся распарсить
    
    # Ищем строку с определением камеры
    # Формат: CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
    for line in cam_lines:
        line = line.strip()
        if line.startswith("#") or not line:
            new_cam_lines.append(line + "\n")
            continue
        
        parts = line.split()
        # Принудительно ставим ID = 1
        parts[0] = "1" 
        new_cam_lines.append(" ".join(parts) + "\n")
        print(f"Fixed camera line: {' '.join(parts)}")

    with open(CAMERAS_FILE, 'w', encoding='utf-8') as f: # 'w' в linux пишет \n
        f.writelines(new_cam_lines)
    print("-> cameras.txt updated (Force ID=1, Unix line endings).")

    # --- ШАГ 2: Исправляем images.txt ---
    if not os.path.exists(IMAGES_FILE):
        print("Error: images.txt not found!")
        return

    with open(IMAGES_FILE, 'r', encoding='utf-8') as f:
        img_lines = f.readlines()

    new_img_lines = []
    is_metadata_line = True # Чередование строк: метаданные -> точки -> метаданные...

    for line in img_lines:
        line = line.strip()
        
        # Пропускаем комментарии
        if line.startswith("#"):
            new_img_lines.append(line + "\n")
            continue
            
        # Если строка пустая, это, вероятно, строка точек (или просто мусор)
        # В COLMAP (txt) строго чередуются: IMAGE_INFO \n POINTS_INFO
        if not line:
            continue

        # Пытаемся понять, это метаданные картинки или точки
        parts = line.split()
        
        # Эвристика: строка метаданных обычно длинная и заканчивается именем файла
        if len(parts) >= 9 and parts[-1].lower().endswith(('.jpg', '.png', '.jpeg')):
            # Это строка метаданных
            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            
            # Принудительно ставим CAMERA_ID (предпоследний элемент) в 1
            parts[-2] = "1"
            
            # Собираем строку обратно
            clean_line = " ".join(parts)
            new_img_lines.append(clean_line + "\n")
            
            # СРАЗУ добавляем пустую строку для точек (так как у нас их нет или мы их не трогаем)
            # Это гарантирует формат "две строки на изображение"
            new_img_lines.append("\n") 
        else:
            # Это строка точек или мусор. Если мы генерируем с нуля из SLAM, 
            # лучше просто пропускать старые точки и ставить пустые строки (как сделано выше),
            # так как gsplat нужны только позы камер.
            pass

    with open(IMAGES_FILE, 'w', encoding='utf-8') as f:
        f.writelines(new_img_lines)
    
    print(f"-> images.txt updated. Processed {len(new_img_lines)//2} images.")

if __name__ == "__main__":
    fix_colmap()