from PIL import Image, UnidentifiedImageError, ImageFile
import numpy as np
from pathlib import Path
import yaml
import os
from multiprocessing import Pool, Manager
from tqdm import tqdm
import warnings
import sys

# --- 安全対策（大きな画像処理での警告無効化） ---
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

########################################
# 実行ファイルがあるディレクトリ（スクリプトまたはexe）を取得
########################################
def get_base_dir():
    if getattr(sys, 'frozen', False):  # PyInstallerでexe化された場合
        return Path(sys.executable).resolve().parent
    else:  # 通常のPythonスクリプトとして実行された場合
        return Path(__file__).resolve().parent


########################################
# 設定読み込み
########################################
def load_config(config_path="setting/default_config.yaml"):
    base_dir = get_base_dir()
    setting_path = base_dir / config_path

    default_config = {
        "tile_images_folder": str(base_dir / "tiles"),
        "rgb_values_file": str(base_dir / "output" / "rgb_values.txt")
    }

    if setting_path.exists():
        with open(setting_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            default_config.update(config)

    # パスを実行ディレクトリ基準に変換（絶対パス保証）
    default_config["tile_images_folder"] = str((base_dir / Path(default_config["tile_images_folder"])).resolve())
    default_config["rgb_values_file"] = str((base_dir / Path(default_config["rgb_values_file"])).resolve())
    default_config["__config_path"] = str(setting_path)  # 設定ファイルのパスも一時保持

    return default_config


########################################
# 設定ファイルへ書き戻し
########################################
def save_config(config):
    config_path = config.get("__config_path")
    if config_path:
        config = {k: v for k, v in config.items() if not k.startswith("__")}  # 内部キー除外
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True)
        print(f"[INFO] 設定ファイルを更新しました → {config_path}")


########################################
# RGB平均値の計算（16分割）
########################################
def calculate_average_colors(image_path_str):
    image_path = Path(image_path_str)
    try:
        img = Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, IOError):
        return None

    width, height = img.size
    sub_width, sub_height = width // 4, height // 4
    avg_colors = []

    for j in range(4):
        for i in range(4):
            left = i * sub_width
            upper = j * sub_height
            right = left + sub_width
            lower = upper + sub_height
            crop = img.crop((left, upper, right, lower))
            np_crop = np.array(crop)
            if np_crop.size == 0:
                avg_colors.append((0, 0, 0))
                continue
            avg_color = np_crop.mean(axis=(0, 1))
            avg_colors.append(tuple(int(x) for x in avg_color))

    return (str(image_path.name), avg_colors)


########################################
# 並列処理ワーカー関数
########################################
def worker(image_path, result_list, progress_queue):
    result = calculate_average_colors(image_path)
    if result:
        result_list.append(result)
    progress_queue.put(1)


########################################
# メイン処理（外部から呼び出せる形式）
########################################
def save_rgb_values_parallel(config=None):
    if config is None:
        config = load_config()

    tile_folder = Path(config["tile_images_folder"])
    output_path = Path(config["rgb_values_file"])

    if not tile_folder.exists():
        print(f"[ERROR] タイル画像フォルダが見つかりません: {tile_folder}")
        return

    image_paths = [str(p) for p in tile_folder.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".gif")]
    if not image_paths:
        print(f"[ERROR] 有効な画像が見つかりません: {tile_folder}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    manager = Manager()
    result_list = manager.list()
    progress_queue = manager.Queue()

    print(f"[INFO] {len(image_paths)} 枚の画像を処理中...")

    with tqdm(total=len(image_paths), desc="RGB計算", unit="枚") as pbar:
        with Pool(processes=os.cpu_count() or 4) as pool:
            for image_path in image_paths:
                pool.apply_async(worker, args=(image_path, result_list, progress_queue))
            for _ in range(len(image_paths)):
                progress_queue.get()
                pbar.update(1)

    with open(output_path, "w", encoding="utf-8") as f:
        for filename, avg_list in result_list:
            f.write(f"{filename}: {avg_list}\n")

    print(f"[INFO] RGB値を保存しました → {output_path}")

    # 保存パスを設定ファイルに記録
    config["rgb_values_file"] = str(output_path)
    save_config(config)


########################################
# スクリプトとしての実行
########################################
def main():
    config = load_config()
    save_rgb_values_parallel(config)

if __name__ == "__main__":
    main()


# from PIL import Image, UnidentifiedImageError, ImageFile
# import numpy as np
# from pathlib import Path
# import yaml
# import os
# from multiprocessing import Pool, Manager
# from tqdm import tqdm
# import warnings
# import sys

# # --- 安全対策（大きな画像処理での警告無効化） ---
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# warnings.simplefilter("ignore", Image.DecompressionBombWarning)

# ########################################
# # 実行ファイルがあるディレクトリ（スクリプトまたはexe）を取得
# ########################################
# def get_base_dir():
#     if getattr(sys, 'frozen', False):  # PyInstallerでexe化された場合
#         return Path(sys.executable).resolve().parent
#     else:  # 通常のPythonスクリプトとして実行された場合
#         return Path(__file__).resolve().parent


# ########################################
# # 設定読み込み
# ########################################
# def load_config(config_path="setting/default_config.yaml"):
#     base_dir = get_base_dir()
#     setting_path = base_dir / config_path

#     default_config = {
#         "tile_images_folder": str(base_dir / "tiles"),
#         "rgb_values_file": str(base_dir / "output" / "rgb_values.txt")
#     }

#     if setting_path.exists():
#         with open(setting_path, "r", encoding="utf-8") as f:
#             config = yaml.safe_load(f) or {}
#             default_config.update(config)

#     # パスを実行ディレクトリ基準に変換（絶対パス保証）
#     default_config["tile_images_folder"] = str((base_dir / Path(default_config["tile_images_folder"])).resolve())
#     default_config["rgb_values_file"] = str((base_dir / Path(default_config["rgb_values_file"])).resolve())

#     return default_config


# ########################################
# # RGB平均値の計算（16分割）
# ########################################
# def calculate_average_colors(image_path_str):
#     image_path = Path(image_path_str)
#     try:
#         img = Image.open(image_path).convert("RGB")
#     except (UnidentifiedImageError, IOError):
#         return None

#     width, height = img.size
#     sub_width, sub_height = width // 4, height // 4
#     avg_colors = []

#     for j in range(4):
#         for i in range(4):
#             left = i * sub_width
#             upper = j * sub_height
#             right = left + sub_width
#             lower = upper + sub_height
#             crop = img.crop((left, upper, right, lower))
#             np_crop = np.array(crop)
#             if np_crop.size == 0:
#                 avg_colors.append((0, 0, 0))
#                 continue
#             avg_color = np_crop.mean(axis=(0, 1))
#             avg_colors.append(tuple(int(x) for x in avg_color))

#     return (str(image_path.name), avg_colors)


# ########################################
# # 並列処理ワーカー関数
# ########################################
# def worker(image_path, result_list, progress_queue):
#     result = calculate_average_colors(image_path)
#     if result:
#         result_list.append(result)
#     progress_queue.put(1)


# ########################################
# # メイン処理（外部から呼び出せる形式）
# ########################################
# def save_rgb_values_parallel(config=None):
#     if config is None:
#         config = load_config()

#     tile_folder = Path(config["tile_images_folder"])
#     output_path = Path(config["rgb_values_file"])

#     if not tile_folder.exists():
#         print(f"[ERROR] タイル画像フォルダが見つかりません: {tile_folder}")
#         return

#     image_paths = [str(p) for p in tile_folder.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".gif")]
#     if not image_paths:
#         print(f"[ERROR] 有効な画像が見つかりません: {tile_folder}")
#         return

#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     manager = Manager()
#     result_list = manager.list()
#     progress_queue = manager.Queue()

#     print(f"[INFO] {len(image_paths)} 枚の画像を処理中...")

#     with tqdm(total=len(image_paths), desc="RGB計算", unit="枚") as pbar:
#         with Pool(processes=os.cpu_count() or 4) as pool:
#             for image_path in image_paths:
#                 pool.apply_async(worker, args=(image_path, result_list, progress_queue))
#             for _ in range(len(image_paths)):
#                 progress_queue.get()
#                 pbar.update(1)

#     with open(output_path, "w", encoding="utf-8") as f:
#         for filename, avg_list in result_list:
#             f.write(f"{filename}: {avg_list}\n")

#     print(f"[INFO] RGB値を保存しました → {output_path}")


# ########################################
# # スクリプトとしての実行
# ########################################
# def main():
#     config = load_config()
#     save_rgb_values_parallel(config)

# if __name__ == "__main__":
#     main()



# # ------------------  rgb_generator.py  ------------------
# from PIL import Image, UnidentifiedImageError
# import numpy as np
# from pathlib import Path
# import yaml
# import os
# from multiprocessing import Pool, Manager
# from tqdm import tqdm

# # === パス設定（main.pyに準拠） ===
# BASE_DIR = Path(__file__).resolve().parent
# SETTING_DIR = BASE_DIR / "setting"
# OUTPUT_DIR = BASE_DIR / "output"
# CONF_FILE = SETTING_DIR / "default_config.yaml"

# # === 設定読み込み ===
# DEFAULT = dict(tile_images_folder=str(BASE_DIR / "tiles"))
# if CONF_FILE.exists():
#     DEFAULT.update(yaml.safe_load(CONF_FILE.read_text(encoding="utf-8")) or {})
# tile_folder = Path(DEFAULT["tile_images_folder"])
# rgb_file = OUTPUT_DIR / "rgb_values.txt"

# # === 並列処理対象関数 ===
# def calculate_average_colors(image_path_str):
#     """ 各画像を16分割し、それぞれの平均RGBを計算 """
#     image_path = Path(image_path_str)
#     try:
#         img = Image.open(image_path).convert("RGB")
#     except (UnidentifiedImageError, IOError):
#         return None  # スキップ

#     width, height = img.size
#     sub_width, sub_height = width // 4, height // 4
#     avg_colors = []

#     for i in range(4):
#         for j in range(4):
#             left = i * sub_width
#             upper = j * sub_height
#             right = left + sub_width
#             lower = upper + sub_height
#             crop = img.crop((left, upper, right, lower))
#             np_crop = np.array(crop)
#             avg_color = np_crop.mean(axis=(0, 1))
#             avg_colors.append(tuple(int(x) for x in avg_color))

#     return (str(image_path.resolve()), avg_colors)

# # === プロセスで実行されるラッパー関数 ===
# def worker(image_path, result_list, progress_queue):
#     result = calculate_average_colors(image_path)
#     if result:
#         result_list.append(result)
#     progress_queue.put(1)

# # === メイン処理 ===
# def save_rgb_values_parallel():
#     if not tile_folder.exists():
#         print(f"ERROR: タイル画像フォルダが見つかりません: {tile_folder}")
#         return

#     image_paths = [str(p) for p in tile_folder.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".gif")]

#     if not image_paths:
#         print(f"ERROR: 有効な画像が見つかりません: {tile_folder}")
#         return

#     OUTPUT_DIR.mkdir(exist_ok=True)
#     open(rgb_file, "w", encoding="utf-8").close()  # 出力ファイル初期化

#     manager = Manager()
#     result_list = manager.list()
#     progress_queue = manager.Queue()

#     print(f"INFO: {len(image_paths)} 枚の画像を処理中...")

#     with tqdm(total=len(image_paths), desc="RGB計算", unit="枚") as pbar:
#         with Pool(processes=os.cpu_count() or 4) as pool:
#             for image_path in image_paths:
#                 pool.apply_async(worker, args=(image_path, result_list, progress_queue))
#             for _ in range(len(image_paths)):
#                 progress_queue.get()
#                 pbar.update(1)

#     # 保存
#     with open(rgb_file, "w", encoding="utf-8") as f:
#         for path, avg_list in result_list:
#             f.write(f"{path}: {avg_list}\n")

#     print(f"INFO: RGB値を保存しました → {rgb_file}")

# # === スクリプト実行 ===
# if __name__ == "__main__":
#     save_rgb_values_parallel()
# # --------------------------------------------------------
