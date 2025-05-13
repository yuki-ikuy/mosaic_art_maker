import os
import yaml
import csv
import time
import warnings
import numpy as np
import ast
import sys
import multiprocessing
from collections import deque
from PIL import Image, ImageEnhance, UnidentifiedImageError
import cv2

warnings.simplefilter("ignore", Image.DecompressionBombWarning)

########################################
# 設定読み込み (default_config.yaml)
########################################
def load_config(config_path="setting/default_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 数値に変換すべき項目
    int_keys = ["num_title", "resize_base_px", "max_tile_usage", "sub_blocks",
                "blur_ksize", "jpeg_quality", "dpi", "repeat_limit",
                "horizontal_limit", "vertical_limit"]
    float_keys = ["brightness_factor", "color_factor"]

    for key in int_keys:
        if key in config:
            config[key] = int(config[key])

    for key in float_keys:
        if key in config:
            config[key] = float(config[key])

    return config

CONFIG = load_config()

# 出力ファイルパスの設定（出力先パスを自動生成 or 固定）
CONFIG["output_path"] = os.path.join("output", "mosaic_output.png")
CONFIG["csv_output_path"] = os.path.join("output", "used.csv")


def resize_with_aspect_ratio(img, base_size):
    """アスペクト比を維持してリサイズ"""
    w, h = img.size
    if w > h:
        new_w = base_size
        new_h = int(h * (base_size / w))
    else:
        new_h = base_size
        new_w = int(w * (base_size / h))
    return img.resize((new_w, new_h), Image.LANCZOS)

def load_rgb_values(file_path):
    """タイル画像の平均RGB値をロード"""
    rgb_values = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if ':' in line:
                name, val = line.split(':', 1)
                name = name.strip()
                avg_rgb = tuple(np.mean(ast.literal_eval(val.strip()), axis=0).astype(int))
                rgb_values[name] = avg_rgb
    return rgb_values

def find_closest_image(target_rgb, rgb_values, used_counts, recent_tiles, x, y, mosaic_map, n):
    """最も近いタイル画像を検索（近隣で未使用のものを優先）"""
    closest_image = None
    closest_distance = float('inf')

    for image_path, rgb in rgb_values.items():
        if used_counts[image_path] < CONFIG["max_tile_usage"] and image_path not in recent_tiles:
            # 縦・横・斜めで同じ画像が使われていないか確認
            if (
                x > 0 and mosaic_map[y * n + (x - 1)] == image_path or
                y > 0 and mosaic_map[(y - 1) * n + x] == image_path or
                x > 0 and y > 0 and mosaic_map[(y - 1) * n + (x - 1)] == image_path
            ):
                continue  # 近隣に同じ画像がある場合はスキップ

            distance = np.linalg.norm(np.array(target_rgb) - np.array(rgb))
            if distance < closest_distance:
                closest_distance = distance
                closest_image = image_path

    return closest_image

def adjust_brightness(tile_img, target_rgb):
    """タイル画像の明るさのみ補正"""
    tile_mean = np.mean(tile_img)
    target_mean = np.mean(target_rgb)

    if tile_mean < 1e-5:
        return tile_img

    factor = target_mean / tile_mean
    return ImageEnhance.Brightness(tile_img).enhance(factor)

def process_block(args):
    """並列処理用のブロック処理"""
    x, y, block_np, rgb_values, used_counts, lock, progress_counter, total_blocks, block_w, block_h, mosaic_map, n = args

    # RGBの平均を計算
    target_rgb = np.mean(block_np, axis=(0, 1)).astype(int)

    # 直近使用したタイルをローカル管理
    recent_tiles = deque(maxlen=10)

    # 最適なタイルを検索
    tile_name = find_closest_image(target_rgb, rgb_values, used_counts, recent_tiles, x, y, mosaic_map, n)
    if tile_name is None:
        return None

    # 画像読み込み & サイズ変更
    tile_img = Image.open(os.path.join(CONFIG["tile_images_folder"], tile_name)).convert("RGB")
    tile_img = tile_img.resize((block_w, block_h))

    # 明るさ補正
    tile_img = adjust_brightness(tile_img, target_rgb)

    # 使用回数を更新（ロックを使って安全にカウント）
    with lock:
        used_counts[tile_name] += 1
        mosaic_map[y * n + x] = tile_name  # モザイクマップに登録
        progress_counter.value += 1  # 進捗を更新
        progress = (progress_counter.value / total_blocks) * 100
        sys.stdout.write(f"\r進捗率: {progress:.2f}% ({progress_counter.value}/{total_blocks})")
        sys.stdout.flush()

    return (x, y, np.array(tile_img, dtype=np.uint8))

def create_mosaic():
    start_time = time.time()

    # 元画像のリサイズ（縦横比を維持）
    base_img = Image.open(CONFIG["original_image_path"])
    base_img = resize_with_aspect_ratio(base_img, CONFIG["resize_base_px"])
    base_img = base_img.convert("RGB")

    w, h = base_img.size
    n = CONFIG["num_title"]
    block_w, block_h = w // n, h // n
    main_np = np.array(base_img)

    # タイル画像のRGBをロード
    rgb_values = load_rgb_values(CONFIG["rgb_values_file"])

    # 共有変数 (タイルの使用回数 & 進捗率管理) - Manager に変更
    with multiprocessing.Manager() as manager:
        used_counts = manager.dict({k: 0 for k in rgb_values.keys()})
        progress_counter = manager.Value("i", 0)  # 進捗カウンター
        lock = manager.Lock()
        mosaic_map = manager.list([None] * (n * n))  # モザイクマップ

        total_blocks = n * n  # 全ブロック数

        # タスクリスト作成
        tasks = [(x, y, main_np[y * block_h:(y + 1) * block_h, x * block_w:(x + 1) * block_w],
                  rgb_values, used_counts, lock, progress_counter, total_blocks, block_w, block_h, mosaic_map, n)
                 for y in range(n) for x in range(n)]

        # 並列処理
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = pool.map(process_block, tasks)

        # 画像の再構築
        mosaic = Image.new("RGB", (w, h))
        for res in results:
            if res:
                x, y, tile_np = res
                mosaic.paste(Image.fromarray(tile_np), (x * block_w, y * block_h))

        # モザイク画像を保存
        mosaic.save(CONFIG["output_path"])
        print(f"\nモザイク画像を保存しました: {CONFIG['output_path']}")

    print(f"\n処理完了: {(time.time() - start_time) / 60:.2f}分")

if __name__ == "__main__":
    create_mosaic()

'''''

import os
import yaml
import csv
import time
import warnings
import numpy as np
import ast
import sys
import multiprocessing
from collections import deque
from PIL import Image, ImageEnhance, UnidentifiedImageError
import cv2

warnings.simplefilter("ignore", Image.DecompressionBombWarning)

########################################
# 設定読み込み (default_config.yaml)
########################################
def load_config(config_path="setting/default_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 数値に変換すべき項目
    int_keys = ["num_title", "resize_base_px", "max_tile_usage", "sub_blocks",
                "blur_ksize", "jpeg_quality", "dpi", "repeat_limit",
                "horizontal_limit", "vertical_limit"]
    float_keys = ["brightness_factor", "color_factor"]

    for key in int_keys:
        if key in config:
            config[key] = int(config[key])

    for key in float_keys:
        if key in config:
            config[key] = float(config[key])

    # 出力ファイルパスのデフォルト指定
    config.setdefault("output_path", os.path.join("output", "mosaic_output.png"))
    config.setdefault("csv_output_path", os.path.join("output", "used.csv"))

    return config


def resize_with_aspect_ratio(img, base_size):
    """アスペクト比を維持してリサイズ"""
    w, h = img.size
    if w > h:
        new_w = base_size
        new_h = int(h * (base_size / w))
    else:
        new_h = base_size
        new_w = int(w * (base_size / h))
    return img.resize((new_w, new_h), Image.LANCZOS)


def load_rgb_values(file_path):
    """タイル画像の平均RGB値をロード"""
    rgb_values = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if ':' in line:
                name, val = line.split(':', 1)
                name = name.strip()
                avg_rgb = tuple(np.mean(ast.literal_eval(val.strip()), axis=0).astype(int))
                rgb_values[name] = avg_rgb
    return rgb_values


def find_closest_image(target_rgb, rgb_values, used_counts, recent_tiles, x, y, mosaic_map, n):
    """最も近いタイル画像を検索（近隣で未使用のものを優先）"""
    closest_image = None
    closest_distance = float('inf')

    for image_path, rgb in rgb_values.items():
        if used_counts[image_path] < used_counts.get("max_tile_usage", 9999) and image_path not in recent_tiles:
            # 縦・横・斜めで同じ画像が使われていないか確認
            if (
                x > 0 and mosaic_map[y * n + (x - 1)] == image_path or
                y > 0 and mosaic_map[(y - 1) * n + x] == image_path or
                x > 0 and y > 0 and mosaic_map[(y - 1) * n + (x - 1)] == image_path
            ):
                continue

            distance = np.linalg.norm(np.array(target_rgb) - np.array(rgb))
            if distance < closest_distance:
                closest_distance = distance
                closest_image = image_path

    return closest_image


def adjust_brightness(tile_img, target_rgb):
    """タイル画像の明るさのみ補正"""
    tile_mean = np.mean(tile_img)
    target_mean = np.mean(target_rgb)

    if tile_mean < 1e-5:
        return tile_img

    factor = target_mean / tile_mean
    return ImageEnhance.Brightness(tile_img).enhance(factor)


def process_block(args):
    """並列処理用のブロック処理"""
    x, y, block_np, rgb_values, used_counts, lock, progress_counter, total_blocks, block_w, block_h, mosaic_map, n, config = args

    target_rgb = np.mean(block_np, axis=(0, 1)).astype(int)
    recent_tiles = deque(maxlen=10)

    tile_name = find_closest_image(target_rgb, rgb_values, used_counts, recent_tiles, x, y, mosaic_map, n)
    if tile_name is None:
        return None

    tile_img = Image.open(os.path.join(config["tile_images_folder"], tile_name)).convert("RGB")
    tile_img = tile_img.resize((block_w, block_h))
    tile_img = adjust_brightness(tile_img, target_rgb)

    with lock:
        used_counts[tile_name] += 1
        mosaic_map[y * n + x] = tile_name
        progress_counter.value += 1
        progress = (progress_counter.value / total_blocks) * 100
        sys.stdout.write(f"\r進捗率: {progress:.2f}% ({progress_counter.value}/{total_blocks})")
        sys.stdout.flush()

    return (x, y, np.array(tile_img, dtype=np.uint8))


def create_mosaic(config):
    start_time = time.time()

    base_img = Image.open(config["original_image_path"])
    base_img = resize_with_aspect_ratio(base_img, config["resize_base_px"])
    base_img = base_img.convert("RGB")

    w, h = base_img.size
    n = config["num_title"]
    block_w, block_h = w // n, h // n
    main_np = np.array(base_img)

    rgb_values = load_rgb_values(config["rgb_values_file"])

    with multiprocessing.Manager() as manager:
        used_counts = manager.dict({k: 0 for k in rgb_values.keys()})
        progress_counter = manager.Value("i", 0)
        lock = manager.Lock()
        mosaic_map = manager.list([None] * (n * n))

        total_blocks = n * n

        tasks = [(x, y, main_np[y * block_h:(y + 1) * block_h, x * block_w:(x + 1) * block_w],
                  rgb_values, used_counts, lock, progress_counter, total_blocks,
                  block_w, block_h, mosaic_map, n, config)
                 for y in range(n) for x in range(n)]

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = pool.map(process_block, tasks)

        mosaic = Image.new("RGB", (w, h))
        for res in results:
            if res:
                x, y, tile_np = res
                mosaic.paste(Image.fromarray(tile_np), (x * block_w, y * block_h))

        mosaic.save(config["output_path"])
        print(f"\nモザイク画像を保存しました: {config['output_path']}")

    print(f"\n処理完了: {(time.time() - start_time) / 60:.2f}分")


# 外部から呼び出す用の関数
def generate_mosaic_image():
    config = load_config()
    create_mosaic(config)


# 単体実行時の処理
if __name__ == "__main__":
    generate_mosaic_image()

'''''
