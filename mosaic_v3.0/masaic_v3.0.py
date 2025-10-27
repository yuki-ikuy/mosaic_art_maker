from __future__ import annotations
import sys, os, shutil, threading, queue, re, yaml, ast, csv, logging, json, pickle
import multiprocessing as mp, warnings, time, math, uuid, random, psutil
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageFile, UnidentifiedImageError, ImageTk
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk
from typing import Optional, List, Tuple
from logging.handlers import RotatingFileHandler

# CustomTkinterの設定
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# カラーパレット（パステル・ソフトトーン）
COLORS = {
    # メインカラー
    'primary_blue': '#87CEEB',        # スカイブルー
    'primary_blue_dark': '#6BB6E8',   # 少し濃いブルー
    'vibrant_pink': '#F8BBD9',        # パステルピンク
    'vibrant_pink_dark': '#F5A3D0',   # 少し濃いピンク
    'teal_green': '#98E4D6',          # パステルティール
    'teal_green_dark': '#85D6C7',     # 少し濃いティール
    'cream_yellow': '#FFF8DC',        # クリーム色
    'cream_yellow_dark': '#F5EFCE',   # 少し濃いクリーム
    
    # 背景とアクセント
    'warm_white': '#FFFEF9',          # ウォームホワイト
    'soft_gray': '#F5F5F5',           # ソフトグレー
    'light_shadow': '#E8E8E8',        # シャドウ
    'text_primary': '#2C3E50',        # ダークグレー
    'text_secondary': '#34495E',      # セカンダリテキスト
    'text_accent': '#27AE60',         # アクセントグリーン
    
    # ホバー効果用
    'hover_lift': '#FFFFFF',          # ホバー時のハイライト
    'hover_shadow': '#D0D0D0',        # ホバー時のシャドウ
    
    # グラデーション補助色
    'gradient_light': '#F0F8FF',      # 極薄ブルー
    'gradient_medium': '#E6F3FF',     # 薄ブルー
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# ─────────────────── 設定パス ────────────────────
def resource(rel: str) -> Path:
    if getattr(sys, 'frozen', False): 
        return Path(sys.executable).parent / rel
    return Path(__file__).resolve().parent / rel

BASE_DIR   = resource("")
SETTING_DIR = BASE_DIR / "settings"; SETTING_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR  = BASE_DIR / "output";  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR   = BASE_DIR / "temp";    TEMP_DIR.mkdir(parents=True, exist_ok=True)
CONF_FILE   = SETTING_DIR / "user_config.yaml"
DETAILED_CONF_FILE = SETTING_DIR / "detailed_settings.yaml"

# デフォルト設定
DEFAULT_CONFIG = {
    'divisions': 100,
    'resize': 10000,
    'max_usage': 2,
    'matching_levels': [1, 2, 4, 8, 16],  # 使用する分割レベル
    'level_weights': {1: 4.0, 2: 3.0, 4: 2.0, 8: 1.5, 16: 1.0},  # 各レベルの重み
    'brightness': 1.0,
    'hue': 1.0,
    'blur': 0,
    'format': "PNG",
    'source_path': "",
    'tile_folder': ""
}

# 16レベルマッチング設定の選択肢
MATCHING_PRESETS = {
    'basic': {
        'name': '基本（1分割）',
        'levels': [1],
        'weights': {1: 1.0},
        'description': '全体の平均色のみで比較。高速処理。'
    },
    'simple': {
        'name': 'シンプル（2分割）', 
        'levels': [1, 2],
        'weights': {1: 3.0, 2: 1.0},
        'description': '全体と左右分割での比較。バランス重視。'
    },
    'standard': {
        'name': 'スタンダード（4分割）',
        'levels': [1, 2, 4],
        'weights': {1: 4.0, 2: 2.0, 4: 1.0},
        'description': '4分割まで使用。品質の向上。'
    },
    'advanced': {
        'name': 'アドバンス（8分割）',
        'levels': [1, 2, 4, 8],
        'weights': {1: 4.0, 2: 3.0, 4: 2.0, 8: 1.0},
        'description': '8分割まで使用。高品質マッチング。'
    },
    'professional': {
        'name': 'プロフェッショナル（16分割）',
        'levels': [1, 2, 4, 8, 16],
        'weights': {1: 4.0, 2: 3.0, 4: 2.0, 8: 1.5, 16: 1.0},
        'description': '16分割フル活用。最高品質マッチング。'
    }
}

# 詳細設定のデフォルト値
DEFAULT_DETAILED_CONFIG = {
    # ランダム選択機能
    'random_selection': False,
    'random_candidates': 3,

    # マッチング精度設定
    'adaptive_matching': True,   # 適応的マッチング
    'color_space': 'RGB',        # RGB, LAB, HSV

    # 高度なオプション
    'cache_size_mb': 512,

    # 顔認識トリミング
    'face_crop': False,          # 顔が必ず入るようにトリミング
    'face_engine': 'auto',       # 'auto' | 'mediapipe' | 'opencv'
    'face_margin_pct': 20,       # 顔周辺余白
    'face_multi_mode': 'all',    # 'all' | 'largest'

    # 作成パターン
    'creation_pattern': 'raster',           # 'raster' | 'snake' | 'random' | 'area_first' | 'spiral'
    'start_area_pct': "0,0,100,100",        # "x1,y1,x2,y2" (%)
    'area_picker_enabled': False,           # プレビューでドラッグ選択
}


# 設定項目の説明文（固定表示用、詳細版）
HELP_TEXTS = {
    'divisions': '画像を分割するブロック数を設定します。数値が大きいほど細かなモザイクになります。',
    'resize': '処理前の画像リサイズサイズを設定します。大きいほど高品質ですが処理時間が長くなります。',
    'max_usage': '同じタイル画像の最大使用回数を制限します。バリエーション豊かなモザイクを作成できます。',
    'matching_preset': '16レベルマッチング精度を選択します。基本は高速、プロフェッショナルは最高品質です。',
    'brightness': 'モザイクアートの明るさを調整します。1.0が標準、2.0で2倍明るくなります。',
    'hue': 'モザイクアートの色の鮮やかさ（彩度）を調整します。1.0が標準値です。',
    'blur': 'モザイクアートにぼかし効果を適用します。0でぼかしなし、10で最大ぼかしです。',
    'random_selection': 'タイル選択にランダム要素を追加して、より自然なモザイクを作成します。',
    'random_candidates': 'ランダム選択時の候補数を設定します。多いほどバリエーション豊かになります。',
    'design_image': 'モザイクアートの元となるデザイン画像を選択してください。JPG、PNG形式に対応しています。',
    'tile_folder': 'モザイクのタイル素材が入ったフォルダを選択してください。多様な画像があるほど美しく仕上がります。',
    'adaptive_matching': '画像の特徴に応じて自動的にマッチング方法を調整します。',
    'color_space': '色の比較に使用する色空間を選択します。LABは人の視覚に近い比較ができます。',
    'cache_size_mb': 'メモリキャッシュのサイズを設定します。大きいほど高速ですがメモリを消費します。',
}

HELP_TEXTS.update({
    'face_crop': 'トリミング時に顔が必ず入るように自動調整します。',
    'face_engine': '顔検出に使うエンジンを選択します。Autoは利用可能な方を自動選択します。',
    'creation_pattern': 'タイルの配置順序を決めます。スネーク/スパイラル/エリア優先などが選べます。',
    'start_area_pct': '「任意エリア優先」時の範囲（%）です。プレビューでドラッグすると自動入力されます。',
    'area_picker_enabled': 'ONにするとプレビュー上でドラッグして開始エリアを設定できます。',
})


LOG_DIR = resource("logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"

# ローテーションするファイルログハンドラ（100MB）
if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    file_handler = RotatingFileHandler(str(LOG_FILE), maxBytes=100*1024*1024, backupCount=100, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

logger.info("ログファイル: %s", LOG_FILE)
# 多重追加防止
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(file_handler)

logger.info("ログファイル: %s", (LOG_DIR / "app.log").resolve())

# ─────── システムリソース監視 ─────────
class ResourceMonitor:
    def __init__(self, cpu_limit=80, memory_limit=90):
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.monitoring = False
        
    def get_usage(self):
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory().percent
        return cpu, memory
    
    def should_throttle(self):
        cpu, memory = self.get_usage()
        return cpu > self.cpu_limit or memory > self.memory_limit

# ───────── マルチレベルRGBリスト生成 ─────────
def calculate_multilevel_colors(path: str):
    """1,2,4,8,16分割でのRGB値を計算"""
    p = Path(path)
    try:
        img = Image.open(p).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        logger.warning("画像読み込み失敗: %s → %s", p.name, e)
        return None
    
    w, h = img.size
    result = {}
    
    # 1分割（全体平均）
    arr = np.asarray(img)
    result[1] = [tuple(int(x) for x in arr.mean(axis=(0,1)))]
    
    # 2分割（左右）
    left = img.crop((0, 0, w//2, h))
    right = img.crop((w//2, 0, w, h))
    result[2] = [
        tuple(int(x) for x in np.asarray(left).mean(axis=(0,1))),
        tuple(int(x) for x in np.asarray(right).mean(axis=(0,1)))
    ]
    
    # 4分割（2×2）
    sw, sh = w//2, h//2
    result[4] = []
    for j in range(2):
        for i in range(2):
            crop = img.crop((i*sw, j*sh, (i+1)*sw, (j+1)*sh))
            arr = np.asarray(crop)
            result[4].append(tuple(int(x) for x in arr.mean(axis=(0,1))) if arr.size else (0,0,0))
    
    # 8分割（2×4 配置）
    sw, sh = w//2, h//4
    result[8] = []
    for j in range(4):
        for i in range(2):
            crop = img.crop((i*sw, j*sh, (i+1)*sw, (j+1)*sh))
            arr = np.asarray(crop)
            result[8].append(tuple(int(x) for x in arr.mean(axis=(0,1))) if arr.size else (0,0,0))
    
    # 16分割（4×4）
    sw, sh = w//4, h//4
    result[16] = []
    for j in range(4):
        for i in range(4):
            crop = img.crop((i*sw, j*sh, (i+1)*sw, (j+1)*sh))
            arr = np.asarray(crop)
            result[16].append(tuple(int(x) for x in arr.mean(axis=(0,1))) if arr.size else (0,0,0))
    
    return p.name, result

def _worker_multilevel_rgb(img_path, res_list, prog_q, skip_q):
    r = calculate_multilevel_colors(img_path)
    if r: res_list.append(r)
    else: skip_q.put(Path(img_path).name)
    prog_q.put(1)

def save_rgb_values_parallel(tile_folder, progress_callback=None):
    from multiprocessing import Pool, Manager
    
    # リソース監視
    monitor = ResourceMonitor()
    
    tile_dir = Path(tile_folder)
    out_path = OUTPUT_DIR / "rgb_values.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [str(p) for p in tile_dir.glob("*") if p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".gif")]
    if len(imgs) < 2:
        raise RuntimeError("タイル画像が2枚未満です")
    
    # CPU使用率に基づいてプロセス数を調整（None対策）
    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 4  # デフォルト値
    
    max_processes = min(cpu_count, len(imgs))
    if monitor.should_throttle():
        max_processes = max(1, max_processes // 2)
    
    m = Manager(); res_list, prog_q, skip_q = m.list(), m.Queue(), m.Queue()
    total = len(imgs)
    
    with Pool(processes=max_processes) as pool:
        for p in imgs: pool.apply_async(_worker_multilevel_rgb, args=(p, res_list, prog_q, skip_q))
        done = 0
        while done < total:
            prog_q.get(); done += 1
            if progress_callback: progress_callback(done, total)
            
            # リソース監視とスロットリング
            if done % 10 == 0 and monitor.should_throttle():
                time.sleep(0.1)
                
        pool.close(); pool.join()
    
    skipped = []
    while not skip_q.empty(): skipped.append(skip_q.get())
    if skipped:
        logger.warning("読み込み失敗画像: %s", ", ".join(skipped))
    
        # マルチレベルRGBデータをJSON形式で保存
    rgb_data = {}
    for fn, levels in res_list:
        # 実ファイル名をキーにする
        rgb_data[fn] = levels

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rgb_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"JSON保存エラー: {str(e)}")
        raise RuntimeError(f"RGB値データの保存中にエラーが発生しました: {str(e)}")
    


def build_processing_order(n: int, pattern: str = "raster", start_area_pct: str = "0,0,100,100"):
    """
    返り値: [(x,y), ...] を常に返す
    pattern: 'raster' | 'snake' | 'random' | 'area_first' | 'spiral'
    """
    import random

    # 基本の格子順序
    order = [(x, y) for y in range(n) for x in range(n)]

    if pattern == "raster":
        return order

    if pattern == "snake":
        snake = []
        for y in range(n):
            row = [(x, y) for x in range(n)]
            if y % 2 == 1:
                row.reverse()
            snake.extend(row)
        return snake

    if pattern == "random":
        rnd = order[:]
        random.shuffle(rnd)
        return rnd

    if pattern == "area_first":
        def parse_area_pct(s):
            try:
                a = [float(t) for t in s.split(",")]
                x1, y1, x2, y2 = (max(0, min(100, v)) for v in a[:4])
                if x1 > x2: x1, x2 = x2, x1
                if y1 > y2: y1, y2 = y2, y1
                return x1, y1, x2, y2
            except Exception:
                return 0.0, 0.0, 100.0, 100.0

        x1, y1, x2, y2 = parse_area_pct(start_area_pct)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        def is_inside(p):
            x, y = p
            # セル中心の%座標
            px = (x + 0.5) * 100.0 / max(1, n)
            py = (y + 0.5) * 100.0 / max(1, n)
            return (x1 <= px <= x2) and (y1 <= py <= y2)

        def dist2(p):
            x, y = p
            px = (x + 0.5) * 100.0 / max(1, n)
            py = (y + 0.5) * 100.0 / max(1, n)
            return (px - cx) ** 2 + (py - cy) ** 2

        inside = [p for p in order if is_inside(p)]
        outside = [p for p in order if not is_inside(p)]
        inside.sort(key=dist2)
        outside.sort(key=dist2)
        return inside + outside

    if pattern == "spiral":
        # 左上から外周→内側の矩形スパイラル（時計回り）
        res = []
        top, left, bottom, right = 0, 0, n - 1, n - 1
        while left <= right and top <= bottom:
            # 左→右
            for x in range(left, right + 1):
                res.append((x, top))
            top += 1
            if top > bottom: break
            # 上→下
            for y in range(top, bottom + 1):
                res.append((right, y))
            right -= 1
            if left > right: break
            # 右→左
            for x in range(right, left - 1, -1):
                res.append((x, bottom))
            bottom -= 1
            if top > bottom: break
            # 下→上
            for y in range(bottom, top - 1, -1):
                res.append((left, y))
            left += 1
        return res

    # 不明パターンはデフォルト
    return order



    
# ───────── モザイク生成─────────
def _resize_keep(img: Image.Image, base: int):
    w, h = img.size
    return img.resize((base, int(h * base / w)), Image.LANCZOS) if w > h \
        else img.resize((int(w * base / h), base), Image.LANCZOS)

def _convert_color_space(rgb_tuple, color_space='RGB'):
    """色空間変換"""
    if color_space == 'RGB':
        return rgb_tuple
    elif color_space == 'LAB':
        # RGB to LAB変換の簡易実装
        r, g, b = [x/255.0 for x in rgb_tuple]
        # XYZ変換
        r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
        g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
        b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92
        x = r * 0.4124 + g * 0.3576 + b * 0.1805
        y = r * 0.2126 + g * 0.7152 + b * 0.0722
        z = r * 0.0193 + g * 0.1192 + b * 0.9505
        return (x, y, z)  # 簡略化
    elif color_space == 'HSV':
        r, g, b = [x/255.0 for x in rgb_tuple]
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        # Hue
        if diff == 0:
            h = 0
        elif max_val == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif max_val == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        else:
            h = (60 * ((r - g) / diff) + 240) % 360
        
        # Saturation
        s = 0 if max_val == 0 else diff / max_val
        
        # Value
        v = max_val
        
        return (h, s*100, v*100)
    
    return rgb_tuple

def _calculate_multilevel_similarity(target_levels: Dict[int, List], tile_levels: Dict[int, List], 
                                   matching_levels: List[int], level_weights: Dict[int, float],
                                   color_space: str = 'RGB'):
    """マルチレベル類似度計算"""
    total_distance = 0
    total_weight = 0
    
    for level in matching_levels:
        if level in target_levels and level in tile_levels:
            target_colors = target_levels[level]
            tile_colors = tile_levels[level]
            
            level_distance = 0
            for t_color, tile_color in zip(target_colors, tile_colors):
                # 色空間変換
                t_converted = _convert_color_space(t_color, color_space)
                tile_converted = _convert_color_space(tile_color, color_space)
                
                # 距離計算
                distance = np.linalg.norm(np.array(t_converted) - np.array(tile_converted))
                level_distance += distance
            
            weight = level_weights.get(level, 1.0)
            total_distance += level_distance * weight
            total_weight += weight * len(target_colors)
    
    return total_distance / total_weight if total_weight > 0 else float('inf')

def _get_block_multilevel_colors(block_arr: np.ndarray) -> Dict[int, List]:
    """ブロック画像のマルチレベル色分析（16レベル対応、2分割は左右）"""
    h, w = block_arr.shape[:2]
    result = {}
    
    # 1分割（全体平均）
    result[1] = [tuple(int(x) for x in block_arr.mean(axis=(0,1)))]
    
    # 2分割（左右）
    left = block_arr[:, :w//2]
    right = block_arr[:, w//2:]
    result[2] = [
        tuple(int(x) for x in left.mean(axis=(0,1))),
        tuple(int(x) for x in right.mean(axis=(0,1)))
    ]
    
    # 4分割（2×2）
    sh, sw = h//2, w//2
    result[4] = []
    for j in range(2):
        for i in range(2):
            section = block_arr[j*sh:(j+1)*sh, i*sw:(i+1)*sw]
            result[4].append(tuple(int(x) for x in section.mean(axis=(0,1))) if section.size else (0,0,0))
    
    # 8分割（2×4）
    sh, sw = h//4, w//2
    result[8] = []
    for j in range(4):
        for i in range(2):
            section = block_arr[j*sh:(j+1)*sh, i*sw:(i+1)*sw]
            result[8].append(tuple(int(x) for x in section.mean(axis=(0,1))) if section.size else (0,0,0))
    
    # 16分割（4×4）
    sh, sw = h//4, w//4
    result[16] = []
    for j in range(4):
        for i in range(4):
            section = block_arr[j*sh:(j+1)*sh, i*sw:(i+1)*sw]
            result[16].append(tuple(int(x) for x in section.mean(axis=(0,1))) if section.size else (0,0,0))
    
    return result

def _find_best_tile_with_constraints(block_levels, rgb_tbl, used_tiles, recent_tiles, 
                                   placement_map, x, y, n, config, detailed_config):
    """制約付きで最適なタイルを検索（16レベル対応）"""
    matching_levels = config.get('matching_levels', [1, 2, 4, 8, 16])
    level_weights = config.get('level_weights', {1: 4.0, 2: 3.0, 4: 2.0, 8: 1.5, 16: 1.0})
    max_usage = config['max_usage']
    random_selection = detailed_config.get('random_selection', False)
    random_candidates = detailed_config.get('random_candidates', 3)
    color_space = detailed_config.get('color_space', 'RGB')
    
    candidates = []
    for tile_name, tile_levels in rgb_tbl.items():
        # 変更点：getで安全に参照
        if used_tiles.get(tile_name, 0) >= max_usage:
            continue

        # 隣接チェック（8方向）
        is_adjacent_used = False
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy == 0 and dx == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n:
                    idx = ny * n + nx
                    # 範囲チェック
                    if 0 <= idx < len(placement_map) and placement_map[idx] == tile_name:
                        is_adjacent_used = True
                        break
            if is_adjacent_used:
                break
        if is_adjacent_used:
            continue

        distance = _calculate_multilevel_similarity(
            block_levels, tile_levels, matching_levels, level_weights, color_space
        )
        candidates.append((tile_name, distance))
    
    if not candidates:
        # 制限を緩める（隣接制限を無視）
        for tile_name, tile_levels in rgb_tbl.items():
            if used_tiles.get(tile_name, 0) >= max_usage:
                continue
            distance = _calculate_multilevel_similarity(
                block_levels, tile_levels, matching_levels, level_weights, color_space
            )
            candidates.append((tile_name, distance))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1])

    if random_selection and len(candidates) > 1:
        selection_count = min(random_candidates, len(candidates))
        selected = random.choice(candidates[:selection_count])
        return selected[0]
    else:
        return candidates[0][0]



class MosaicProgress:
    """モザイク作成の進行状況を管理"""
    def __init__(self, temp_file: Path):
        self.temp_file = temp_file
        self.data = {
            'completed_blocks': [],
            'used_tiles': {},
            'current_position': 0,
            'total_blocks': 0,
            'placement_map': [],
            'paused': False
        }
        self.load()
    
    def save(self):
        with open(self.temp_file, 'wb') as f:
            pickle.dump(self.data, f)
    
    def load(self):
        if self.temp_file.exists():
            try:
                with open(self.temp_file, 'rb') as f:
                    self.data.update(pickle.load(f))
            except:
                pass
    
    def clear(self):
        if self.temp_file.exists():
            self.temp_file.unlink()
        self.data = {
            'completed_blocks': [],
            'used_tiles': {},
            'current_position': 0,
            'total_blocks': 0,
            'placement_map': [],
            'paused': False
        }



def create_mosaic(config, detailed_config, progress_callback=None, pause_check=None):
    import gc

    monitor = ResourceMonitor(memory_limit=90)

    # ── 入力検証・基本値 ──
    n = int(config.get('divisions', DEFAULT_CONFIG['divisions']))
    MAX_TILES = 300
    if n <= 0:
        raise ValueError("分割数が不正です。1以上を指定してください。")
    if n > MAX_TILES:
        raise RuntimeError(f"分割数が多すぎます（上限{MAX_TILES}）")

    base = Path(config['source_path'])
    if not base.is_file():
        raise RuntimeError("デザイン画像が見つかりません")

    tile_dir = Path(config['tile_folder'])
    if not tile_dir.is_dir():
        raise RuntimeError("タイルフォルダが見つかりません")

    rgb_json = OUTPUT_DIR / "rgb_values.json"
    if not rgb_json.is_file():
        raise RuntimeError("マルチレベルRGB値ファイルが見つかりません")

    # ── 進行状況ファイルを先に作る ──
    temp_file = TEMP_DIR / f"mosaic_progress_{uuid.uuid4().hex[:8]}.pkl"
    mosaic_progress = MosaicProgress(temp_file)
    mosaic_progress.data['paused'] = False

    # ── タイル色テーブル読み込み（キーを int に復元）──
    with open(rgb_json, 'r', encoding='utf-8') as f:
        rgb_tbl_raw = json.load(f)
    rgb_tbl = {}
    for name, levels in rgb_tbl_raw.items():
        fixed = {}
        for k, v in levels.items():
            try:
                fixed[int(k)] = v
            except Exception:
                continue
        rgb_tbl[name] = fixed

    # ── 元画像読み込み → 顔トリミング（任意）→ リサイズ ──
    src_img = Image.open(base).convert("RGB")
    if detailed_config.get('face_crop', False):
        face_engine = str(detailed_config.get('face_engine', 'auto'))
        margin = int(detailed_config.get('face_margin_pct', 20))
        multi  = str(detailed_config.get('face_multi_mode', 'all'))
        target_aspect = (src_img.width / src_img.height) if src_img.height else 1.0
        src_img = _face_aware_crop(
            src_img, target_aspect,
            face_engine=face_engine, multi_mode=multi, margin_pct=margin
        )
    img = _resize_keep(src_img, int(config.get('resize', DEFAULT_CONFIG['resize'])))

    w, h = img.size
    bw, bh = w // n, h // n
    if bw == 0 or bh == 0:
        raise ValueError("分割数が大きすぎます。画像サイズに対して '分割数' を小さくしてください。")

    total = n * n
    mosaic_progress.data['total_blocks'] = total

    # ── 使用回数／配置マップの復元 ──
    used_tiles_saved = mosaic_progress.data.get('used_tiles', {})
    used_tiles = {k: int(used_tiles_saved.get(k, 0)) for k in rgb_tbl.keys()}

    placement_map = mosaic_progress.data.get('placement_map')
    if not isinstance(placement_map, list) or len(placement_map) != total:
        placement_map = [None] * total
        mosaic_progress.data['placement_map'] = placement_map

    # ── これまでの配置を画像に反映（再開時）──
    mosaic = Image.new("RGB", (w, h))
    for i, tile_name in enumerate(placement_map):
        if tile_name:
            x = i % n
            y = i // n
            left  = x * bw
            upper = y * bh
            right = (x + 1) * bw if x < n - 1 else w
            lower = (y + 1) * bh if y < n - 1 else h
            tile_path = tile_dir / tile_name
            if tile_path.exists():
                tile = Image.open(tile_path).convert("RGB").resize((right-left, lower-upper))
                mosaic.paste(tile, (left, upper))

    # ── ブロック走査順の構築 ──
    recent_tiles = deque(maxlen=10)
    pattern = str(detailed_config.get('creation_pattern', 'raster'))
    start_area_pct = detailed_config.get('start_area_pct', "0,0,100,100")
    order = build_processing_order(n, pattern, start_area_pct)

    # 進捗初期値
    blocks_done = sum(1 for v in placement_map if v is not None)
    if progress_callback:
        progress_callback(blocks_done, total)

    # ── 本処理 ──
    for (x, y) in order:
        i = y * n + x
        if placement_map[i] is not None:
            continue

        if pause_check and pause_check():
            mosaic_progress.data['current_position'] = i
            mosaic_progress.data['paused'] = True
            mosaic_progress.save()
            return None  # 一時停止

        left  = x * bw
        upper = y * bh
        right = (x + 1) * bw if x < n - 1 else w
        lower = (y + 1) * bh if y < n - 1 else h

        block = img.crop((left, upper, right, lower))
        block_arr = np.asarray(block)
        block_levels = _get_block_multilevel_colors(block_arr)

        best_name = _find_best_tile_with_constraints(
            block_levels, rgb_tbl, used_tiles, recent_tiles,
            placement_map, x, y, n, config, detailed_config
        )

        if best_name:
            used_tiles[best_name] = used_tiles.get(best_name, 0) + 1
            placement_map[i] = best_name
            recent_tiles.append(best_name)

            tile_path = tile_dir / best_name
            if tile_path.exists():
                tile = Image.open(tile_path).convert("RGB").resize((right-left, lower-upper))
                mosaic.paste(tile, (left, upper))

            # 進行状況を更新・保存
            mosaic_progress.data['completed_blocks'].append({'x': x, 'y': y, 'tile': best_name})
            mosaic_progress.data['used_tiles'] = used_tiles
            mosaic_progress.data['placement_map'] = placement_map

            blocks_done += 1
            if progress_callback:
                progress_callback(blocks_done, total)

        # メモリ負荷が高ければクールダウン
        if blocks_done % 50 == 0 and monitor.should_throttle():
            gc.collect()
            time.sleep(0.1)

        del block_arr

    # ── エフェクト ──
    if config['brightness'] != 1.0:
        mosaic = ImageEnhance.Brightness(mosaic).enhance(config['brightness'])
    if config['hue'] != 1.0:
        mosaic = ImageEnhance.Color(mosaic).enhance(config['hue'])
    if config['blur'] > 0:
        k = int(config['blur']) | 1
        mosaic = mosaic.filter(ImageFilter.GaussianBlur(radius=k))

    # ── 保存 ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config['format'].upper() == "JPEG":
        out_file = OUTPUT_DIR / f"mosaic_{ts}.jpg"
        mosaic.save(out_file, "JPEG", quality=95)
    else:
        out_file = OUTPUT_DIR / f"mosaic_{ts}.png"
        mosaic.save(out_file, "PNG")

    # CSV出力
    out_csv = OUTPUT_DIR / f"used_tiles_{ts}.csv"
    try:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "used_count"])
            for tile_name, count in sorted(used_tiles.items()):
                if count > 0:
                    writer.writerow([tile_name, count])
    except Exception as e:
        logger.error("CSV保存時エラー: %s", e)

    mosaic_progress.clear()
    logger.info("モザイク保存 → %s", out_file.resolve())
    logger.info("CSV保存 → %s", out_csv.resolve())
    return out_file, out_csv


def _detect_faces_mediapipe(pil_img: Image.Image) -> List[Tuple[int,int,int,int]]:
    """人の顔のみ（MediaPipe）。返り値: [(x,y,w,h), ...]。未インストールなら空配列。"""
    try:
        import mediapipe as mp
        import numpy as np, cv2
    except Exception:
        return []

    img = np.array(pil_img)  # RGB
    h, w = img.shape[:2]
    try:
        mp_fd = mp.solutions.face_detection
        with mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            res = fd.process(bgr)
            boxes = []
            if res.detections:
                for det in res.detections:
                    bb = det.location_data.relative_bounding_box
                    x = int(bb.xmin * w)
                    y = int(bb.ymin * h)
                    ww = int(bb.width * w)
                    hh = int(bb.height * h)
                    if ww > 0 and hh > 0:
                        boxes.append((max(0, x), max(0, y), ww, hh))
            return boxes
    except Exception:
        return []

def _detect_faces_opencv(pil_img: Image.Image) -> List[Tuple[int,int,int,int]]:  # CHANGED: 引数簡略化
    """OpenCV Haar（人の顔）。OpenCV内蔵の既定カスケードを自動利用。"""
    try:
        import cv2, numpy as np, os
    except Exception:
        return []

    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)

    try:
        human = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    except Exception:
        return []

    if human is None or human.empty():
        return []

    faces = human.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def _face_aware_crop(pil_img: Image.Image, target_aspect: float,
                     face_engine: str = 'auto',
                     multi_mode: str = 'all', margin_pct: int = 20) -> Image.Image:
    """
    顔が必ず入るように target_aspect を厳守してクロップ。
    画像の引き伸ばし/圧縮は一切せず、可能な限り“広い”領域を切り取る。
    """
    img_w, img_h = pil_img.size
    rects: List[Tuple[int,int,int,int]] = []

    try_mediapipe = face_engine in ('auto', 'mediapipe')
    try_opencv    = face_engine in ('auto', 'opencv')

    if try_mediapipe:
        rects += _detect_faces_mediapipe(pil_img)
    if try_opencv and not rects:
        rects += _detect_faces_opencv(pil_img)

    if not rects:
        # 顔が無い場合は元画像を target_aspect で最大面積クロップ
        return _max_area_crop_with_aspect(pil_img, target_aspect)

    def _union_rects(rs):
        x1 = min([x for (x, y, w, h) in rs])
        y1 = min([y for (x, y, w, h) in rs])
        x2 = max([x + w for (x, y, w, h) in rs])
        y2 = max([y + h for (x, y, w, h) in rs])
        return (x1, y1, x2 - x1, y2 - y1)

    def _largest_rect(rs):
        return max(rs, key=lambda r: r[2] * r[3])

    base = _largest_rect(rects) if multi_mode == 'largest' else _union_rects(rects)

    # マージン（余白）を足す
    x, y, w, h = base
    mx = int(w * margin_pct / 100)
    my = int(h * margin_pct / 100)
    x -= mx; y -= my; w += 2 * mx; h += 2 * my
    x = max(0, x); y = max(0, y)
    w = min(img_w - x, w); h = min(img_h - y, h)

    # 1) 顔(＋余白)を必ず含む"最小"のトリミング寸法（aspect厳守）
    #    target_aspect = Wc/Hc
    Wc_min = max(w, int(math.ceil(h * target_aspect)))
    Hc_min = max(h, int(math.ceil(w / target_aspect)))

    # 最小寸法は aspect を満たすように再調整
    if Wc_min / Hc_min < target_aspect:
        Wc_min = int(math.ceil(Hc_min * target_aspect))
    else:
        Hc_min = int(math.ceil(Wc_min / target_aspect))

    # 2) そこから画像内で許される最大まで等倍拡大（引き伸ばしなし）
    s_max = min(img_w / Wc_min, img_h / Hc_min)
    s_max = max(1.0, s_max)  # 最低1.0（縮小はしない＝“最大を選ぶ”）
    W = min(img_w, int(Wc_min * s_max))
    H = min(img_h, int(Hc_min * s_max))

    # aspect厳守で端の丸め誤差調整
    if W / H > target_aspect:
        W = int(H * target_aspect)
    else:
        H = int(W / target_aspect)
    W = max(1, min(W, img_w)); H = max(1, min(H, img_h))

    # 3) 位置決め：顔(＋余白)を含むように、かつ画像内で最大限センタリング
    #    設置可能な x1 は [face_x2 - W, face_x1] ∩ [0, img_w - W]
    fx1, fy1, fw, fh = x, y, w, h
    fx2, fy2 = fx1 + fw, fy1 + fh

    x1_min = max(0, fx2 - W)
    x1_max = min(img_w - W, fx1)
    y1_min = max(0, fy2 - H)
    y1_max = min(img_h - H, fy1)

    # 顔中心に近い位置を優先（可能範囲で近いほう）
    fcx, fcy = fx1 + fw / 2, fy1 + fh / 2
    prefer_x1 = int(round(fcx - W / 2))
    prefer_y1 = int(round(fcy - H / 2))

    def _clamp_pref(pref, lo, hi):
        if lo > hi:
            return max(0, min(img_w - W, pref)) if W else 0
        return max(lo, min(hi, pref))

    x1 = _clamp_pref(prefer_x1, x1_min, x1_max)
    y1 = _clamp_pref(prefer_y1, y1_min, y1_max)

    return pil_img.crop((x1, y1, x1 + W, y1 + H))


def _max_area_crop_with_aspect(pil_img: Image.Image, target_aspect: float) -> Image.Image:
    """顔が見つからない場合に、画像全体から target_aspect を満たす最大領域を切り出す。"""
    img_w, img_h = pil_img.size
    if img_w == 0 or img_h == 0:
        return pil_img

    # 画像全体から aspect を守って最大
    if (img_w / img_h) > target_aspect:
        # 横が余る → 高さに合わせて幅を決める
        H = img_h
        W = int(round(H * target_aspect))
    else:
        W = img_w
        H = int(round(W / target_aspect))

    x1 = (img_w - W) // 2
    y1 = (img_h - H) // 2
    return pil_img.crop((x1, y1, x1 + W, y1 + H))


# ─────────────────── パスのポータブル保存/復元（NEW） ───────────────────
def to_portable_path(p: Optional[str]) -> str:
    """
    絶対パスを ${BASE}/ 相対 or ${HOME}/ 相対のトークン表現へ変換。
    """
    if not p:
        return ""
    p = os.path.normpath(p)
    base = str(BASE_DIR)
    home = os.path.expanduser("~")
    try:
        if os.path.commonpath([p, base]) == base:
            rel = os.path.relpath(p, base)
            return f"${{BASE}}/{rel}".replace("\\", "/")
    except Exception:
        pass
    try:
        if os.path.commonpath([p, home]) == home:
            rel = os.path.relpath(p, home)
            return f"${{HOME}}/{rel}".replace("\\", "/")
    except Exception:
        pass
    # 最後の手段としてそのまま返す（なるべく避けたいが互換のため）
    return p.replace("\\", "/")

def from_portable_path(s: Optional[str]) -> Optional[str]:
    """
    トークンを実パスに展開。存在しなければ None。
    """
    if not s:
        return None
    s = s.replace("\\", "/")
    if s.startswith("${BASE}/"):
        p = BASE_DIR / s[len("${BASE}/"):]
    elif s.startswith("${HOME}/"):
        p = Path(os.path.expanduser("~")) / s[len("${HOME}/"):]
    else:
        p = Path(s)
    p = p.resolve()
    return str(p) if p.exists() else None

def try_resolve_or_search(s: Optional[str], filename_only: bool = False) -> Optional[str]:
    """
    1) from_portable_path で解決
    2) 見つからなければ、同名ファイルを BASE_DIR 配下から探索（軽い再リンク用）
    """
    p = from_portable_path(s)
    if p:
        return p
    if not s:
        return None
    name = os.path.basename(s) if not filename_only else s
    for root, dirs, files in os.walk(BASE_DIR):
        if name in files or name in dirs:
            cand = Path(root) / name
            return str(cand.resolve())
    return None


# ────────── GUI ──────────
class MosaicMaker:
    def __init__(self):
        # メインウィンドウの設定
        self.root = ctk.CTk()
        self.root.title("Mosaic Art Maker ver3.0")
        self.root.geometry("1920x1080")
        self.root.minsize(1200, 800)
        
        # 初期化フラグ
        self.widgets_created = False
        self.initialization_complete = False
        
        # 変数の初期化
        self.source_image = None
        self.source_image_path = None
        self.tile_folder_path = None
        self.is_processing = False
        self.zoom_level = 1.0
        self.image_position = [0, 0]
        self.is_dragging = False
        self.drag_start = [0, 0]
        self.canvas_image = None
        self.canvas_image_id = None
        
        # 処理状態管理
        self.rgb_state = "idle"  # idle, running, completed
        self.mosaic_state = "idle"  # idle, running, paused, completed
        self.pause_flag = threading.Event()
        
        # UI モード管理（基本設定をデフォルトに変更）
        self.ui_mode = "basic"  # basic, advanced
        
        # ウィジェット参照を初期化
        self.basic_frame = None
        self.advanced_frame = None
        self.tile_count_label = None
        self.image_size_label = None
        self.estimated_time_label = None
        self.estimated_size_label = None
        self.help_label = None
        self.progress_bar = None
        self.progress_label = None
        self.rgb_button = None
        self.mosaic_button = None
        self.pause_button = None
        self.source_button = None
        self.tile_button = None

        self.area_select_active = False
        self.area_rect_screen = None 
        self.area_rect_id = None
        self._display_geom = None
        
        self.fit_on_next_draw = True

        # 設定値（16レベル対応）
        self.basic_settings = {
            'divisions': ctk.IntVar(value=DEFAULT_CONFIG['divisions']),
            'resize': ctk.IntVar(value=DEFAULT_CONFIG['resize']),
            'max_usage': ctk.IntVar(value=DEFAULT_CONFIG['max_usage']),
            'brightness': ctk.DoubleVar(value=DEFAULT_CONFIG['brightness']),
            'hue': ctk.DoubleVar(value=DEFAULT_CONFIG['hue']),
            'blur': ctk.IntVar(value=DEFAULT_CONFIG['blur']),
            'format': ctk.StringVar(value=DEFAULT_CONFIG['format'])
        }
        
        # 詳細設定値（16レベル対応）
        self.advanced_settings = {
            'matching_preset': ctk.StringVar(value="basic"),  # デフォルトはスタンダード
            'random_selection': ctk.BooleanVar(value=DEFAULT_DETAILED_CONFIG['random_selection']),
            'random_candidates': ctk.IntVar(value=DEFAULT_DETAILED_CONFIG['random_candidates']),
            'adaptive_matching': ctk.BooleanVar(value=DEFAULT_DETAILED_CONFIG['adaptive_matching']),
            'color_space': ctk.StringVar(value=DEFAULT_DETAILED_CONFIG['color_space']),
            'cache_size_mb': ctk.IntVar(value=DEFAULT_DETAILED_CONFIG['cache_size_mb']),
        }

        self.advanced_settings.update({
            'face_crop': tk.BooleanVar(value=DEFAULT_DETAILED_CONFIG['face_crop']),
            'face_engine': tk.StringVar(value=DEFAULT_DETAILED_CONFIG['face_engine']),
            'face_margin_pct': tk.IntVar(value=DEFAULT_DETAILED_CONFIG['face_margin_pct']),
            'face_multi_mode': tk.StringVar(value=DEFAULT_DETAILED_CONFIG['face_multi_mode']),
            'creation_pattern': tk.StringVar(value=DEFAULT_DETAILED_CONFIG['creation_pattern']),
            'start_area_pct': tk.StringVar(value=DEFAULT_DETAILED_CONFIG['start_area_pct']),
            'area_picker_enabled': tk.BooleanVar(value=DEFAULT_DETAILED_CONFIG['area_picker_enabled']),
        })

                
        # 現在の説明文を保持
        self.current_help_text = "Mosaic Art Maker - ようこそ！"
        
        # 安全な初期化
        self.safe_initialize()
    
    def create_header_title(self, parent):
        """アプリのタイトル（左側）"""
        title_wrap = ctk.CTkFrame(parent, fg_color="transparent")
        title_wrap.pack(side="left", padx=10)

        ctk.CTkLabel(
            title_wrap,
            text="Mosaic Art Maker",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(anchor="w")
    
    def safe_initialize(self):
        """安全な初期化処理"""
        try:
            logger.info("ウィジェット作成を開始...")
            self.create_widgets()
            self.widgets_created = True
            logger.info("ウィジェット作成完了")
            
            logger.info("イベントバインディングを設定...")
            self.setup_bindings()
            logger.info("イベントバインディング完了")
            
            logger.info("設定ファイルを読み込み...")
            self.load_config()
            logger.info("設定ファイル読み込み完了")
            
            self.initialization_complete = True
            logger.info("初期化が正常に完了しました")
            
        except Exception as e:
            logger.error("初期化エラー: %s", e)
            self.create_error_ui(str(e))

    
    def create_error_ui(self, error_message):
        """エラー時の最小限UI"""
        try:
            for widget in self.root.winfo_children():
                widget.destroy()
                
            error_frame = ctk.CTkFrame(self.root)
            error_frame.pack(fill="both", expand=True, padx=50, pady=50)
            
            ctk.CTkLabel(
                error_frame,
                text="初期化エラー",
                font=ctk.CTkFont(size=24, weight="bold"),
                text_color="red"
            ).pack(pady=20)
            
            ctk.CTkLabel(
                error_frame,
                text=f"エラー詳細: {error_message}",
                font=ctk.CTkFont(size=14),
                wraplength=800
            ).pack(pady=10)
            
            ctk.CTkButton(
                error_frame,
                text="再起動",
                font=ctk.CTkFont(size=16, weight="bold"),
                command=self.restart_application
            ).pack(pady=30)
            
        except Exception as e2:
            logger.error("エラーUI作成失敗: %s", e2)
    
    def restart_application(self):
        """アプリケーション再起動"""
        try:
            self.root.destroy()
            import sys
            import os
            os.execv(sys.executable, ['python'] + sys.argv)
        except Exception as e:
            logger.error("再起動失敗: %s", e)
    
    def create_modern_button(self, parent, text, command=None, width=None, height=45, 
                           primary_color=None, hover_color=None, text_size=16, **kwargs):
        """立体感のある現代的なボタンを作成"""
        try:
            if primary_color is None:
                primary_color = COLORS['primary_blue']
            if hover_color is None:
                hover_color = COLORS['primary_blue_dark']
                
            button_kwargs = {
                'text': text,
                'command': command,
                'height': height,
                'corner_radius': 20,
                'fg_color': primary_color,
                'hover_color': hover_color,
                'text_color': COLORS['text_primary']
            }
            
            if width is not None:
                button_kwargs['width'] = width
                
            try:
                button_kwargs['font'] = ctk.CTkFont(size=text_size, weight="bold")
            except Exception:
                pass
                
            try:
                button_kwargs['border_width'] = 2
                button_kwargs['border_color'] = COLORS['light_shadow']
            except Exception:
                pass
                
            button_kwargs.update(kwargs)
            button = ctk.CTkButton(parent, **button_kwargs)
            
            # ホバー効果
            try:
                original_color = primary_color
                
                def on_enter(event):
                    try:
                        if button.winfo_exists():
                            button.configure(fg_color=hover_color)
                    except Exception:
                        pass
                
                def on_leave(event):
                    try:
                        if button.winfo_exists():
                            button.configure(fg_color=original_color)
                    except Exception:
                        pass
                
                button.bind("<Enter>", on_enter)
                button.bind("<Leave>", on_leave)
            except Exception:
                pass
            
            return button
            
        except Exception as e:
            logger.error("ボタン作成エラー: %s", e)
            return ctk.CTkButton(parent, text=text, command=command, width=width or 200, height=height)
    
    def create_widgets(self):
        """メインウィジェット作成"""
        self.create_header_section()
        self.create_main_content_section()
        self.create_footer_section()
    
    def create_header_section(self):
        """ヘッダーセクション作成"""
        self.header_frame = ctk.CTkFrame(
            self.root, 
            height=90, 
            corner_radius=0, 
            fg_color=COLORS['gradient_light']
        )
        self.header_frame.pack(fill="x", padx=0, pady=0)
        self.header_frame.pack_propagate(False)
        
        header_content = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        header_content.pack(fill="both", expand=True, padx=20, pady=15)
        
        self.create_header_title(header_content)
        self.create_header_mode_switcher(header_content)
        self.create_header_help_area(header_content)
    
    def create_header_mode_switcher(self, parent):
        """ヘッダーモード切り替え + 設定リセット/保存ボタン"""
        mode_frame = ctk.CTkFrame(parent, fg_color="transparent")
        mode_frame.pack(side="left", expand=True, padx=20)

        mode_label = ctk.CTkLabel(
            mode_frame,
            text="表示モード:",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS['text_primary']
        )
        mode_label.pack()

        mode_buttons = ctk.CTkFrame(mode_frame, fg_color="transparent")
        mode_buttons.pack(pady=5)

        # 表示モード切替
        self.basic_mode_btn = self.create_modern_button(
            mode_buttons,
            text="基本設定",
            width=120,
            height=35,
            primary_color=COLORS['vibrant_pink'],
            hover_color=COLORS['vibrant_pink_dark'],
            text_size=12,
            command=lambda: self.switch_ui_mode("basic")
        )
        self.basic_mode_btn.pack(side="left", padx=5)

        self.advanced_mode_btn = self.create_modern_button(
            mode_buttons,
            text="詳細設定",
            width=120,
            height=35,
            primary_color=COLORS['teal_green'],
            hover_color=COLORS['teal_green_dark'],
            text_size=12,
            command=lambda: self.switch_ui_mode("advanced")
        )
        self.advanced_mode_btn.pack(side="left", padx=5)

        # 設定リセット（既存のヘッダーリセットを使っていた場合もこれでOK）
        self.reset_header_button = self.create_modern_button(
            mode_buttons,
            text="設定リセット",
            width=120,
            height=35,
            primary_color=COLORS['primary_blue'],
            hover_color=COLORS['primary_blue_dark'],
            text_size=12,
            command=self.reset_to_defaults
        )
        self.reset_header_button.pack(side="left", padx=12)

        # 設定保存（ヘッダーに配置）
        self.save_header_button = self.create_modern_button(
            mode_buttons,
            text="設定保存",
            width=120,
            height=35,
            primary_color=COLORS['text_accent'],          # 落ち着いたグリーン
            hover_color=COLORS['teal_green_dark'],        # 近いトーンでホバー
            text_size=12,
            command=self.manual_save_config
        )
        self.save_header_button.pack(side="left", padx=5)

    
    def create_header_help_area(self, parent):
        """ヘッダーヘルプエリア作成"""
        self.help_frame = ctk.CTkFrame(
            parent, 
            fg_color=COLORS['warm_white'],
            corner_radius=15,
            border_width=2,
            border_color=COLORS['light_shadow']
        )
        self.help_frame.pack(side="right", fill="both", expand=True)
        
        self.help_label = ctk.CTkLabel(
            self.help_frame,
            text=self.current_help_text,
            font=ctk.CTkFont(size=16),
            text_color=COLORS['text_primary'],
            wraplength=700,
            justify="left"
        )
        self.help_label.pack(expand=True, padx=15, pady=10)
    
    def create_main_content_section(self):
        """メインコンテンツセクション作成"""
        self.main_frame = ctk.CTkFrame(
            self.root, 
            corner_radius=0,
            fg_color=COLORS['warm_white']
        )
        self.main_frame.pack(fill="both", expand=True, padx=0, pady=0)
        
        # 左パネル（設定エリア）
        self.left_panel = ctk.CTkFrame(
            self.main_frame, 
            width=650,
            fg_color=COLORS['warm_white']
        )
        self.left_panel.pack(side="left", fill="y", padx=15, pady=15)
        self.left_panel.pack_propagate(False)
        
        # 右パネル（プレビューエリア）
        self.right_panel = ctk.CTkFrame(
            self.main_frame,
            fg_color=COLORS['warm_white']
        )
        self.right_panel.pack(side="right", fill="both", expand=True, padx=15, pady=15)
        
        self.create_left_panel_content()
        self.create_right_panel_content()
    
    def create_footer_section(self):
        """フッターセクション作成"""
        self.footer_frame = ctk.CTkFrame(
            self.root, 
            height=60, 
            corner_radius=0,
            fg_color=COLORS['gradient_medium']
        )
        self.footer_frame.pack(fill="x", padx=0, pady=0)
        self.footer_frame.pack_propagate(False)
        
        footer_content = ctk.CTkFrame(self.footer_frame, fg_color="transparent")
        footer_content.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.create_footer_progress_section(footer_content)
    
    def create_left_panel_content(self):
        """左パネル内容作成"""
        # スクロール可能フレーム
        self.scrollable_frame = ctk.CTkScrollableFrame(
            self.left_panel,
            fg_color=COLORS['warm_white']
        )
        self.scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.create_execute_control_section()
        self.create_basic_settings_frame()
        self.create_advanced_settings_frame()
        
        # 初期状態は基本設定を表示
        self.switch_ui_mode("basic")
    
    def create_execute_control_section(self):
        """実行コントロールセクション作成"""
        self.execute_frame = ctk.CTkFrame(
            self.left_panel,
            fg_color=COLORS['text_accent'],
            corner_radius=20,
            border_width=2,
            border_color=COLORS['light_shadow']
        )
        self.execute_frame.pack(fill="x", padx=10, pady=10)
        
        execute_header = ctk.CTkLabel(
            self.execute_frame,
            text="実行コントロール",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="white"
        )
        execute_header.pack(pady=(10, 5))
        
        execute_container = ctk.CTkFrame(self.execute_frame, fg_color="transparent")
        execute_container.pack(fill="x", padx=15, pady=10)
        
        self.create_execute_buttons(execute_container)
        self.create_status_label(execute_container)
    
    def create_execute_buttons(self, parent):
        """実行ボタン作成"""
        # RGB生成ボタンのみ
        self.rgb_button = self.create_modern_button(
            parent,
            text="RGBリスト生成開始",
            width=None,
            height=40,
            primary_color=COLORS['teal_green'],
            hover_color=COLORS['teal_green_dark'],
            text_size=12,
            command=self.generate_rgb_list
        )
        self.rgb_button.pack(fill="x", pady=3)

        # モザイク生成
        self.mosaic_button = self.create_modern_button(
            parent,
            text="モザイクアート作成開始",
            width=None,
            height=45,
            primary_color=COLORS['vibrant_pink'],
            hover_color=COLORS['vibrant_pink_dark'],
            text_size=14,
            command=self.toggle_mosaic
        )
        self.mosaic_button.pack(fill="x", pady=3)

    
    def create_status_label(self, parent):
        """ステータスラベル作成"""
        self.status_label = ctk.CTkLabel(
            parent, 
            text="準備完了 - 設定を確認して実行してください",
            font=ctk.CTkFont(size=10),
            text_color="white"
        )
        self.status_label.pack(pady=5)
    
    def create_basic_settings_frame(self):
        """基本設定フレーム作成"""
        self.basic_frame = ctk.CTkFrame(
            self.scrollable_frame,
            fg_color=COLORS['warm_white']
        )
        
        # ファイル選択セクション
        file_section = self.create_section_header(
            self.basic_frame, "ファイル選択", 
            COLORS['cream_yellow'], COLORS['text_primary']
        )
        
        self.create_file_selection_basic(file_section)
        
        # 基本パラメータセクション
        self.create_collapsible_section(
            self.basic_frame, "基本設定", COLORS['primary_blue'],
            lambda parent: self.create_basic_params(parent)
        )
        
        # エフェクトセクション
        self.create_collapsible_section(
            self.basic_frame, "エフェクト設定", COLORS['vibrant_pink'],
            lambda parent: self.create_effect_params(parent)
        )
    
    def create_file_selection_basic(self, parent):
        """基本設定用ファイル選択UI"""
        # デザイン画像選択
        design_frame = ctk.CTkFrame(parent, fg_color="transparent")
        design_frame.pack(fill="x", padx=15, pady=8)
        
        ctk.CTkLabel(
            design_frame, 
            text="デザイン画像:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(anchor="w", pady=(0, 5))
        
        self.source_button = self.create_modern_button(
            design_frame, 
            text="画像ファイルを選択してください", 
            width=None,
            height=35,
            primary_color=COLORS['vibrant_pink'],
            hover_color=COLORS['vibrant_pink_dark'],
            text_size=11,
            command=self.select_source_image
        )
        self.source_button.pack(fill="x", pady=3)
        self.add_help_binding(self.source_button, 'design_image')
        
        # タイル画像フォルダ選択
        tile_frame = ctk.CTkFrame(parent, fg_color="transparent")
        tile_frame.pack(fill="x", padx=15, pady=8)
        
        ctk.CTkLabel(
            tile_frame, 
            text="タイル画像フォルダ:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(anchor="w", pady=(0, 5))
        
        self.tile_button = self.create_modern_button(
            tile_frame, 
            text="タイル素材フォルダを選択してください", 
            width=None,
            height=35,
            primary_color=COLORS['teal_green'],
            hover_color=COLORS['teal_green_dark'],
            text_size=11,
            command=self.select_tile_folder
        )
        self.tile_button.pack(fill="x", pady=3)
        self.add_help_binding(self.tile_button, 'tile_folder')
    
    def create_advanced_settings_frame(self):
        """詳細設定フレーム作成"""
        self.advanced_frame = ctk.CTkFrame(
            self.scrollable_frame,
            fg_color=COLORS['warm_white']
        )
        
        # ファイル選択（参照表示）
        file_section = self.create_section_header(
            self.advanced_frame, "ファイル選択", 
            COLORS['cream_yellow'], COLORS['text_primary']
        )
        
        self.create_file_selection_advanced(file_section)
        
        # 16レベルマッチング設定（既存）
        self.create_collapsible_section(
            self.advanced_frame, "16マッチング精度", COLORS['teal_green'],
            lambda parent: self.create_matching_params(parent)
        )

        # ランダム選択設定（既存）
        self.create_collapsible_section(
            self.advanced_frame, "ランダム選択", COLORS['vibrant_pink'],
            lambda parent: self.create_random_params(parent)
        )

        # 色空間・高度設定（既存）
        self.create_collapsible_section(
            self.advanced_frame, "色空間・高度設定", COLORS['cream_yellow'],
            lambda parent: self.create_color_space_params(parent)
        )

        # --- 顔認識トリミング ---
        self.create_collapsible_section(
            self.advanced_frame, "顔認識トリミング(実験的機能)", COLORS['primary_blue'],
            lambda parent: self.create_face_crop_params(parent)
        )

        # ---作成パターン ---
        self.create_collapsible_section(
            self.advanced_frame, "作成パターン", COLORS['cream_yellow'],
            lambda parent: self.create_pattern_params(parent)
        )

        # エフェクト設定（詳細版／既存）
        self.create_collapsible_section(
            self.advanced_frame, "エフェクト設定", COLORS['vibrant_pink'],
            lambda parent: self.create_effect_params(parent)
        )

    def create_file_selection_advanced(self, parent):
        """詳細設定用ファイル選択UI"""
        info_frame = ctk.CTkFrame(
            parent,
            fg_color=COLORS['cream_yellow'],
            corner_radius=10,
            border_width=1,
            border_color=COLORS['light_shadow']
        )
        info_frame.pack(fill="x", padx=15, pady=8)
        
        ctk.CTkLabel(
            info_frame,
            text="ファイル選択は実行コントロール下部で行ってください",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(pady=10)
        
        status_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
        status_frame.pack(fill="x", padx=15, pady=5)
        
        # 現在の選択状況を表示
        self.source_status_label = ctk.CTkLabel(
            status_frame,
            text="デザイン画像: 未選択",
            font=ctk.CTkFont(size=11),
            text_color=COLORS['text_secondary']
        )
        self.source_status_label.pack(anchor="w", pady=2)
        
        self.tile_status_label = ctk.CTkLabel(
            status_frame,
            text="タイル素材: 未選択",
            font=ctk.CTkFont(size=11),
            text_color=COLORS['text_secondary']
        )
        self.tile_status_label.pack(anchor="w", pady=2)
    
    def create_right_panel_content(self):
        """右パネル内容作成"""
        self.create_preview_section()
        self.create_statistics_section()
    
    def create_preview_section(self):
        """プレビューセクション作成"""
        preview_frame = ctk.CTkFrame(
            self.right_panel,
            fg_color=COLORS['primary_blue'],
            corner_radius=20,
            border_width=2,
            border_color=COLORS['light_shadow']
        )
        preview_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.create_preview_header(preview_frame)
        self.create_preview_canvas(preview_frame)
    
    def create_preview_header(self, parent):
        """プレビューヘッダー作成"""
        preview_header = ctk.CTkFrame(parent, height=60, fg_color="transparent")
        preview_header.pack(fill="x", padx=15, pady=10)
        preview_header.pack_propagate(False)
        
        header_left = ctk.CTkFrame(preview_header, fg_color="transparent")
        header_left.pack(side="left", fill="y")
        
        ctk.CTkLabel(
            header_left, 
            text="リアルタイムプレビュー", 
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="white"
        ).pack(anchor="w")
        
        ctk.CTkLabel(
            header_left, 
            text="マウスホイール: ズーム | ドラッグ: 移動", 
            font=ctk.CTkFont(size=10),
            text_color=COLORS['soft_gray']
        ).pack(anchor="w")
        
        self.create_preview_controls(preview_header)
    
    def create_preview_controls(self, parent):
        """プレビューコントロール（フィット削除済み）"""
        control_frame = ctk.CTkFrame(parent, fg_color="transparent")
        control_frame.pack(side="right", fill="y")
        
        # ズーム情報
        self.zoom_label = ctk.CTkLabel(
            control_frame, 
            text="100%",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="white"
        )
        self.zoom_label.pack(side="top", pady=2)
        
        # コントロールボタン
        button_row = ctk.CTkFrame(control_frame, fg_color="transparent")
        button_row.pack(side="top")
        
        self.reset_view_button = self.create_modern_button(
            button_row,
            text="リセット",
            width=60,
            height=25,
            primary_color=COLORS['vibrant_pink'],
            hover_color=COLORS['vibrant_pink_dark'],
            text_size=10,
            command=self.reset_view
        )
        self.reset_view_button.pack(side="left", padx=2)

    def create_preview_canvas(self, parent):
        """プレビューキャンバス作成"""
        canvas_container = ctk.CTkFrame(
            parent,
            fg_color=COLORS['warm_white'],
            corner_radius=15,
            border_width=1,
            border_color=COLORS['light_shadow']
        )
        canvas_container.pack(fill="both", expand=True, padx=15, pady=10)
        
        # 画像キャンバス
        self.canvas = ctk.CTkCanvas(
            canvas_container, 
            bg=COLORS['cream_yellow'], 
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_statistics_section(self):
        """統計情報セクション作成"""
        stats_frame = ctk.CTkFrame(
            self.right_panel,
            fg_color=COLORS['teal_green'],
            corner_radius=20,
            border_width=2,
            border_color=COLORS['light_shadow']
        )
        stats_frame.pack(fill="x", padx=10, pady=10)
        
        # 統計ヘッダー
        stats_header = ctk.CTkFrame(stats_frame, fg_color="transparent")
        stats_header.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(
            stats_header, 
            text="統計情報", 
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="white"
        ).pack()
        
        # 統計情報グリッド
        self.create_statistics_grid(stats_frame)
    
    def create_statistics_grid(self, parent):
        """統計情報グリッド作成"""
        stats_container = ctk.CTkFrame(parent, fg_color="transparent")
        stats_container.pack(padx=15, pady=10)
        
        stats_grid = ctk.CTkFrame(
            stats_container,
            fg_color=COLORS['warm_white'],
            corner_radius=15,
            border_width=1,
            border_color=COLORS['light_shadow']
        )
        stats_grid.pack(fill="x")
        
        # 統計ラベル
        self.tile_count_label = ctk.CTkLabel(
            stats_grid, 
            text="タイル数: 計算中...",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS['text_primary']
        )
        self.tile_count_label.grid(row=0, column=0, padx=15, pady=8, sticky="w")
        
        self.image_size_label = ctk.CTkLabel(
            stats_grid, 
            text="画像サイズ: 計算中...",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS['text_primary']
        )
        self.image_size_label.grid(row=0, column=1, padx=15, pady=8, sticky="w")
        

    def create_footer_progress_section(self, parent):
        """フッター進行状況セクション作成"""
        progress_section = ctk.CTkFrame(parent, fg_color="transparent")
        progress_section.pack(fill="x", pady=2)
        
        progress_header = ctk.CTkLabel(
            progress_section, 
            text="処理進行状況",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS['text_primary']
        )
        progress_header.pack(side="left")
        
        progress_controls = ctk.CTkFrame(progress_section, fg_color="transparent")
        progress_controls.pack(side="left", fill="x", expand=True, padx=(15, 0))
        
        self.progress_bar = ctk.CTkProgressBar(
            progress_controls, 
            width=400,
            height=20,
            progress_color=COLORS['vibrant_pink'],
            corner_radius=10,
            border_width=1,
            border_color=COLORS['light_shadow']
        )
        self.progress_bar.pack(side="left", fill="x", expand=True)
        self.progress_bar.set(0)
        
        self.progress_label = ctk.CTkLabel(
            progress_controls, 
            text="0 / 0 (0%)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS['text_primary']
        )
        self.progress_label.pack(side="left", padx=10)
    
    def create_footer_buttons(self, parent):
        """フッター右側ボタン（削除）"""
        # 何も作らない（右下の「保存」ボタンを非表示）
        return

    
    def create_section_header(self, parent, title, bg_color, text_color="white"):
        """セクションヘッダーを作成"""
        section_frame = ctk.CTkFrame(
            parent,
            fg_color=bg_color,
            corner_radius=15,
            border_width=2,
            border_color=COLORS['light_shadow']
        )
        section_frame.pack(fill="x", padx=8, pady=8)
        
        header_label = ctk.CTkLabel(
            section_frame, 
            text=title, 
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=text_color
        )
        header_label.pack(pady=10)
        
        return section_frame
    
    def create_collapsible_section(self, parent, title, bg_color, content_creator):
        """折りたたみ可能なセクションを作成"""
        section_frame = ctk.CTkFrame(
            parent,
            fg_color=bg_color,
            corner_radius=15,
            border_width=2,
            border_color=COLORS['light_shadow']
        )
        section_frame.pack(fill="x", padx=8, pady=8)
        
        # ヘッダー（クリック可能）
        header_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=5)
        
        # 展開/折りたたみ状態を管理
        is_expanded = tk.BooleanVar(value=True)
        
        toggle_button = self.create_modern_button(
            header_frame,
            text=f"▼ {title}",
            width=None,
            height=30,
            primary_color=bg_color,
            hover_color=bg_color,
            text_size=12,
            command=lambda: self.toggle_section(section_frame, is_expanded, toggle_button, title, content_frame)
        )
        toggle_button.pack(fill="x")
        
        # コンテンツフレーム
        content_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
        content_frame.pack(fill="x", padx=5, pady=5)
        
        # コンテンツを作成
        content_creator(content_frame)
        
        return section_frame
    
    def toggle_section(self, section_frame, is_expanded, toggle_button, title, content_frame):
        """セクションの展開/折りたたみを切り替え"""
        if is_expanded.get():
            content_frame.pack_forget()
            toggle_button.configure(text=f"▶ {title}")
            is_expanded.set(False)
        else:
            content_frame.pack(fill="x", padx=5, pady=5)
            toggle_button.configure(text=f"▼ {title}")
            is_expanded.set(True)
    
    def create_basic_params(self, parent):
        """基本パラメータの内容を作成"""
        self.create_compact_slider_entry_pair(parent, "分割数:", 'divisions', 50, 300, 10)
        self.create_compact_slider_entry_pair(parent, "リサイズ (px):", 'resize', 5000, 30000, 1000)
        self.create_compact_slider_entry_pair(parent, "最大使用数:", 'max_usage', 1, 10, 1)
    
    def create_effect_params(self, parent):
        """エフェクトパラメータの内容を作成"""
        self.create_compact_slider_entry_pair(parent, "明るさ:", 'brightness', 0.3, 2.0, 0.1, is_float=True)
        self.create_compact_slider_entry_pair(parent, "色の彩度:", 'hue', 0.3, 2.0, 0.1, is_float=True)
        self.create_compact_slider_entry_pair(parent, "ぼかし効果:", 'blur', 0, 10, 1)
        
        # 出力形式
        self.create_format_selection(parent)
    
    def create_format_selection(self, parent):
        """出力形式選択UI作成"""
        format_frame = ctk.CTkFrame(
            parent,
            fg_color=COLORS['warm_white'],
            corner_radius=10,
            border_width=1,
            border_color=COLORS['light_shadow']
        )
        format_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            format_frame, 
            text="出力形式:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(pady=(8, 5))
        
        format_radio_frame = ctk.CTkFrame(format_frame, fg_color="transparent")
        format_radio_frame.pack(pady=(0, 8))
        
        ctk.CTkRadioButton(
            format_radio_frame,
            text="PNG ",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS['text_primary'],
            variable=self.basic_settings['format'],
            value="PNG",
            radiobutton_width=16,
            radiobutton_height=16,
            fg_color=COLORS['primary_blue']
        ).pack(side="left", padx=20)
        
        ctk.CTkRadioButton(
            format_radio_frame,
            text="JPEG",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS['text_primary'],
            variable=self.basic_settings['format'],
            value="JPEG",
            radiobutton_width=16,
            radiobutton_height=16,
            fg_color=COLORS['primary_blue']
        ).pack(side="left", padx=20)
    
    def create_matching_params(self, parent):
        """16レベルマッチング設定（ラジオボタン形式）"""
        matching_inner = ctk.CTkFrame(
            parent, 
            fg_color=COLORS['warm_white'],
            corner_radius=10,
            border_width=1,
            border_color=COLORS['light_shadow']
        )
        matching_inner.pack(fill="x", padx=10, pady=10)
        
        # 説明ラベル
        info_label = ctk.CTkLabel(
            matching_inner,
            text="マッチング精度レベルを選択してください：",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS['text_primary']
        )
        info_label.pack(pady=(15, 10))
        
        # ラジオボタンフレーム
        radio_frame = ctk.CTkFrame(matching_inner, fg_color="transparent")
        radio_frame.pack(fill="x", padx=15, pady=10)
        
        # ラジオボタンを作成
        self.matching_radio_buttons = {}
        for preset_key, preset_data in MATCHING_PRESETS.items():
            # ラジオボタン行
            button_row = ctk.CTkFrame(radio_frame, fg_color="transparent")
            button_row.pack(fill="x", pady=5)
            
            # ラジオボタン
            radio_btn = ctk.CTkRadioButton(
                button_row,
                text=preset_data['name'],
                variable=self.advanced_settings['matching_preset'],
                value=preset_key,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=COLORS['text_primary'],
                radiobutton_width=18,
                radiobutton_height=18,
                fg_color=COLORS['primary_blue'],
                hover_color=COLORS['primary_blue_dark'],
                command=self.update_weight_display
            )
            radio_btn.pack(side="left", padx=10)
            
            # ヘルプバインディングを追加
            self.add_help_binding(radio_btn, 'matching_preset')
            
            # 説明ラベル
            desc_label = ctk.CTkLabel(
                button_row,
                text=preset_data['description'],
                font=ctk.CTkFont(size=10),
                text_color=COLORS['text_secondary']
            )
            desc_label.pack(side="left", padx=(10, 0))
            
            self.matching_radio_buttons[preset_key] = radio_btn
        
        # 重み表示エリア
        self.create_weight_display_area(matching_inner)
    
    def create_weight_display_area(self, parent):
        """重み表示エリア作成"""
        weight_frame = ctk.CTkFrame(
            parent,
            fg_color=COLORS['soft_gray'],
            corner_radius=10,
            border_width=1,
            border_color=COLORS['light_shadow']
        )
        weight_frame.pack(fill="x", padx=15, pady=15)
        
        weight_header = ctk.CTkLabel(
            weight_frame,
            text="選択されたレベルの重み設定:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS['text_primary']
        )
        weight_header.pack(pady=(10, 5))
        
        # 重み表示ラベル（動的更新）
        self.weight_display_label = ctk.CTkLabel(
            weight_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=COLORS['text_secondary'],
            wraplength=500
        )
        self.weight_display_label.pack(pady=(0, 10), padx=15)
    
    def create_random_params(self, parent):
        """ランダム選択設定"""
        random_inner = ctk.CTkFrame(
            parent,
            fg_color=COLORS['warm_white'],
            corner_radius=10,
            border_width=1,
            border_color=COLORS['light_shadow']
        )
        random_inner.pack(fill="x", padx=10, pady=10)
        
        random_check = ctk.CTkCheckBox(
            random_inner,
            text="ランダム選択機能を有効にする",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS['text_primary'],
            variable=self.advanced_settings['random_selection']
        )
        random_check.pack(pady=12, padx=15)
        self.add_help_binding(random_check, 'random_selection')
        
        # 候補数設定
        self.create_advanced_slider_entry_pair(random_inner, "ランダム候補数:", 'random_candidates', 2, 10, 1)
    
    def create_color_space_params(self, parent):
        """色空間・高度設定"""
        color_inner = ctk.CTkFrame(
            parent,
            fg_color=COLORS['warm_white'],
            corner_radius=10,
            border_width=1,
            border_color=COLORS['light_shadow']
        )
        color_inner.pack(fill="x", padx=10, pady=10)
        
        # 色空間選択
        self.create_color_space_selection(color_inner)
        
        # 適応的マッチング
        ctk.CTkCheckBox(
            color_inner,
            text="適応的マッチング",
            variable=self.advanced_settings['adaptive_matching'],
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(anchor="w", padx=15, pady=5)
    
    def create_color_space_selection(self, parent):
        """色空間選択UI作成"""
        color_frame = ctk.CTkFrame(parent, fg_color="transparent")
        color_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(
            color_frame, 
            text="色空間:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(anchor="w", pady=(0, 5))
        
        color_radio_frame = ctk.CTkFrame(color_frame, fg_color="transparent")
        color_radio_frame.pack()
        
        color_spaces = [
            ("RGB (標準)", "RGB"),
            ("LAB (知覚的)", "LAB"),
            ("HSV (色相)", "HSV")
        ]
        
        for text, value in color_spaces:
            ctk.CTkRadioButton(
                color_radio_frame,
                text=text,
                variable=self.advanced_settings['color_space'],
                value=value,
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=COLORS['text_primary']
            ).pack(anchor="w", padx=10, pady=2)
    
    def create_compact_slider_entry_pair(self, parent, label_text, var_name, min_val, max_val, step, is_float=False):
        """コンパクトなスライダーと数値入力のペア"""
        container_frame = ctk.CTkFrame(parent, fg_color="transparent")
        container_frame.pack(fill="x", padx=15, pady=5)
        
        inner_frame = ctk.CTkFrame(
            container_frame, 
            fg_color=COLORS['warm_white'],
            corner_radius=10,
            border_width=1,
            border_color=COLORS['light_shadow']
        )
        inner_frame.pack(fill="x")
        
        # ラベルと入力欄のフレーム
        label_frame = ctk.CTkFrame(inner_frame, fg_color="transparent")
        label_frame.pack(fill="x", padx=10, pady=8)
        
        label = ctk.CTkLabel(
            label_frame, 
            text=label_text,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS['text_primary']
        )
        label.pack(side="left")
        
        # 浮動小数点値の場合、表示フォーマットを制御
        entry = self.create_value_entry(label_frame, var_name, is_float)
        entry.pack(side="right")
        
        # スライダー
        slider = self.create_value_slider(inner_frame, var_name, min_val, max_val, step, is_float)
        slider.pack(fill="x", padx=10, pady=(0, 10))
        
        # ヘルプバインディング
        self.add_help_binding(entry, var_name)
        self.add_help_binding(slider, var_name)
        self.add_help_binding(label, var_name)
        
        # 値の範囲チェック
        self.add_value_validation(var_name, min_val, max_val)
    
    def create_value_entry(self, parent, var_name, is_float):
        """値入力エントリー作成"""
        if is_float:
            # フォーマット用の文字列変数を作成
            display_var = tk.StringVar()
            
            def update_display(*args):
                try:
                    value = self.basic_settings[var_name].get()
                    display_var.set(f"{value:.1f}")
                except:
                    pass
            
            # 初期値設定
            update_display()
            self.basic_settings[var_name].trace_add('write', update_display)
            
            entry = ctk.CTkEntry(
                parent, 
                width=80, 
                height=25,
                textvariable=display_var,
                font=ctk.CTkFont(size=11, weight="bold"),
                corner_radius=8,
                border_width=1,
                border_color=COLORS['light_shadow'],
                fg_color=COLORS['soft_gray']
            )
            
            # エントリー変更時に元の変数を更新
            def on_entry_change(*args):
                try:
                    value = float(display_var.get())
                    self.basic_settings[var_name].set(value)
                except:
                    pass
            
            display_var.trace_add('write', on_entry_change)
        else:
            entry = ctk.CTkEntry(
                parent, 
                width=80, 
                height=25,
                textvariable=self.basic_settings[var_name],
                font=ctk.CTkFont(size=11, weight="bold"),
                corner_radius=8,
                border_width=1,
                border_color=COLORS['light_shadow'],
                fg_color=COLORS['soft_gray']
            )
        
        return entry
    
    def create_value_slider(self, parent, var_name, min_val, max_val, step, is_float):
        """値スライダー作成"""
        if is_float:
            steps = int((max_val - min_val) / step)
            slider = ctk.CTkSlider(
                parent, 
                from_=min_val, 
                to=max_val,
                number_of_steps=steps,
                variable=self.basic_settings[var_name],
                progress_color=COLORS['vibrant_pink'],
                button_color=COLORS['teal_green'],
                button_hover_color=COLORS['teal_green_dark'],
                height=16
            )
        else:
            steps = int((max_val - min_val) / step)
            slider = ctk.CTkSlider(
                parent, 
                from_=min_val, 
                to=max_val, 
                number_of_steps=steps,
                variable=self.basic_settings[var_name],
                progress_color=COLORS['vibrant_pink'],
                button_color=COLORS['teal_green'],
                button_hover_color=COLORS['teal_green_dark'],
                height=16
            )
        
        return slider
    
    def add_value_validation(self, var_name, min_val, max_val):
        """値の範囲チェック追加"""
        def validate_range(*args):
            try:
                if not self.widgets_created:
                    return
                    
                value = self.basic_settings[var_name].get()
                if value is None:
                    if var_name in DEFAULT_CONFIG:
                        self.basic_settings[var_name].set(DEFAULT_CONFIG[var_name])
                    return
                    
                if value < min_val:
                    self.basic_settings[var_name].set(min_val)
                elif value > max_val:
                    self.basic_settings[var_name].set(max_val)
                    
                if hasattr(self, 'root') and self.root:
                    self.root.after_idle(self.update_statistics)
            except Exception as e:
                logger.warning("値検証エラー (%s): %s", var_name, e)
                if var_name in DEFAULT_CONFIG:
                    try:
                        self.basic_settings[var_name].set(DEFAULT_CONFIG[var_name])
                    except:
                        pass
        
        self.basic_settings[var_name].trace_add('write', validate_range)
    
    def create_advanced_slider_entry_pair(self, parent, label_text, var_name, min_val, max_val, step):
        """詳細設定用のコンパクトなスライダー"""
        container_frame = ctk.CTkFrame(parent, fg_color="transparent")
        container_frame.pack(fill="x", padx=15, pady=5)
        
        inner_frame = ctk.CTkFrame(
            container_frame, 
            fg_color=COLORS['warm_white'],
            corner_radius=10,
            border_width=1,
            border_color=COLORS['light_shadow']
        )
        inner_frame.pack(fill="x")
        
        # ラベルと入力欄のフレーム
        label_frame = ctk.CTkFrame(inner_frame, fg_color="transparent")
        label_frame.pack(fill="x", padx=10, pady=8)
        
        label = ctk.CTkLabel(
            label_frame, 
            text=label_text,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS['text_primary']
        )
        label.pack(side="left")
        
        entry = ctk.CTkEntry(
            label_frame, 
            width=80, 
            height=25,
            textvariable=self.advanced_settings[var_name],
            font=ctk.CTkFont(size=11, weight="bold"),
            corner_radius=8,
            border_width=1,
            border_color=COLORS['light_shadow'],
            fg_color=COLORS['soft_gray']
        )
        entry.pack(side="right")
        
        # スライダー
        steps = int((max_val - min_val) / step)
        slider = ctk.CTkSlider(
            inner_frame, 
            from_=min_val, 
            to=max_val, 
            number_of_steps=steps,
            variable=self.advanced_settings[var_name],
            progress_color=COLORS['vibrant_pink'],
            button_color=COLORS['teal_green'],
            button_hover_color=COLORS['teal_green_dark'],
            height=16
        )
        slider.pack(fill="x", padx=10, pady=(0, 10))
        
        # ヘルプバインディング
        self.add_help_binding(entry, var_name)
        self.add_help_binding(slider, var_name)
        self.add_help_binding(label, var_name)
    
    def add_help_binding(self, widget, help_key):
        """ヘルプテキストバインディング"""
        if not widget or help_key not in HELP_TEXTS:
            return
            
        try:
            def on_enter(event):
                try:
                    if help_key in HELP_TEXTS and self.help_label and self.help_label.winfo_exists():
                        self.current_help_text = HELP_TEXTS[help_key]
                        self.help_label.configure(text=self.current_help_text)
                except Exception as e:
                    logger.debug("ヘルプテキスト表示エラー: %s", e)
            
            def on_focus(event):
                try:
                    if help_key in HELP_TEXTS and self.help_label and self.help_label.winfo_exists():
                        self.current_help_text = HELP_TEXTS[help_key]
                        self.help_label.configure(text=self.current_help_text)
                except Exception as e:
                    logger.debug("ヘルプテキスト表示エラー: %s", e)
            
            widget.bind("<Enter>", on_enter)
            widget.bind("<FocusIn>", on_focus)
        except Exception as e:
            logger.warning("ヘルプバインディング設定エラー: %s", e)
    
    def switch_ui_mode(self, mode):
        """UIモードを切り替え"""
        if not self.widgets_created:
            return
            
        try:
            self.ui_mode = mode
            
            if mode == "basic":
                if hasattr(self, 'advanced_frame') and self.advanced_frame:
                    self.advanced_frame.pack_forget()
                if hasattr(self, 'basic_frame') and self.basic_frame:
                    self.basic_frame.pack(fill="both", expand=True)
                    
                self.basic_mode_btn.configure(
                    fg_color=COLORS['vibrant_pink'],
                    hover_color=COLORS['vibrant_pink_dark']
                )
                self.advanced_mode_btn.configure(
                    fg_color=COLORS['teal_green'],
                    hover_color=COLORS['teal_green_dark']
                )
                self.current_help_text = "基本設定モード: 初心者向けの基本的な設定のみ表示されています。"
            else:
                if hasattr(self, 'basic_frame') and self.basic_frame:
                    self.basic_frame.pack_forget()
                if hasattr(self, 'advanced_frame') and self.advanced_frame:
                    self.advanced_frame.pack(fill="both", expand=True)
                    
                self.advanced_mode_btn.configure(
                    fg_color=COLORS['vibrant_pink'],
                    hover_color=COLORS['vibrant_pink_dark']
                )
                self.basic_mode_btn.configure(
                    fg_color=COLORS['teal_green'],
                    hover_color=COLORS['teal_green_dark']
                )
                self.current_help_text = "詳細設定モード: 16レベルマッチングや高度な設定が利用できます。"
            
            if self.help_label and self.help_label.winfo_exists():
                self.help_label.configure(text=self.current_help_text)
                
            # 詳細モードに切り替えた時は重み表示を更新
            if mode == "advanced" and hasattr(self, 'update_weight_display'):
                self.root.after(100, self.update_weight_display)
        except Exception as e:
            logger.error("UIモード切り替えエラー: %s", e)
    
    def update_weight_display(self):
        """選択されたマッチングプリセットに応じて重み表示を更新"""
        try:
            preset_key = self.advanced_settings['matching_preset'].get()
            if preset_key in MATCHING_PRESETS:
                preset_data = MATCHING_PRESETS[preset_key]
                levels = preset_data['levels']
                weights = preset_data['weights']
                
                # 重み表示文字列を生成
                weight_text = "使用分割: " + ", ".join(f"{level}" for level in levels) + "\n"
                weight_text += "重み: " + ", ".join(f"{level}={weights[level]}" for level in levels)
                
                if hasattr(self, 'weight_display_label') and self.weight_display_label:
                    self.weight_display_label.configure(text=weight_text)
                    
                self.update_log(f"マッチング設定変更: {preset_data['name']}")
        except Exception as e:
            logger.debug("重み表示更新エラー: %s", e)
    
    def setup_bindings(self):
        """イベントバインディングを設定（キーボードショートカット追加）"""
        if not self.widgets_created:
            return
            
        try:
            # キーボードショートカット
            self.root.bind('<Control-s>', lambda e: self.manual_save_config())
            self.root.bind('<Control-o>', lambda e: self.select_source_image())
            self.root.bind('<Control-d>', lambda e: self.select_tile_folder())
            self.root.bind('<Control-r>', lambda e: self.reset_to_defaults())
            self.root.bind('<F1>', lambda e: self.show_help_dialog())
            self.root.bind('<F5>', lambda e: self.refresh_preview())
            
            # キャンバスのマウスイベント
            if hasattr(self, 'canvas') and self.canvas:
                self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
                self.canvas.bind("<Button-1>", self.on_mouse_down)
                self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
                self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
                # 右クリックでリセット
                self.canvas.bind("<Button-3>", lambda e: self.reset_view())
            
            # 設定変更時のプレビュー更新と自動保存
            for var_name in ['brightness', 'hue', 'blur']:
                if var_name in self.basic_settings:
                    self.basic_settings[var_name].trace_add('write', lambda *args: self.safe_refresh_preview())
                    self.basic_settings[var_name].trace_add('write', lambda *args: self.safe_save_config())
                
            # 統計更新のバインディング
            for var_name in ['divisions', 'resize']:
                if var_name in self.basic_settings:
                    self.basic_settings[var_name].trace_add('write', lambda *args: self.safe_update_statistics())
                    self.basic_settings[var_name].trace_add('write', lambda *args: self.safe_save_config())
        except Exception as e:
            logger.error("イベントバインディング設定エラー: %s", e)
    
    def safe_refresh_preview(self):
        """安全なプレビュー更新"""
        try:
            if self.widgets_created and hasattr(self, 'refresh_preview'):
                self.refresh_preview()
        except Exception as e:
            logger.debug("プレビュー更新エラー: %s", e)
            
    def safe_save_config(self):
        """安全な設定保存"""
        try:
            if self.widgets_created and hasattr(self, 'save_config'):
                self.save_config()
        except Exception as e:
            logger.debug("設定保存エラー: %s", e)
            
    def safe_update_statistics(self):
        """安全な統計更新"""
        try:
            if self.widgets_created and hasattr(self, 'update_statistics'):
                self.update_statistics()
        except Exception as e:
            logger.debug("統計更新エラー: %s", e)
    
    def show_help_dialog(self):
        """ヘルプダイアログを表示"""
        help_text = """Mosaic Art Maker ver3.0

主な機能:
• 16レベル高精度マッチング
• 基本設定/詳細設定の段階表示
• リアルタイムプレビュー
• 一時停止・再開機能

キーボードショートカット:
• Ctrl+S: 設定保存
• Ctrl+O: デザイン画像選択
• Ctrl+D: タイル素材フォルダ選択
• Ctrl+R: 設定リセット
• F1: このヘルプ表示
• F5: プレビュー更新
• Esc: 処理一時停止
• 右クリック: プレビューリセット

マウス操作（プレビュー）:
• ホイール: ズーム
• ドラッグ: 画像移動
• 右クリック: 表示リセット

Tips:
• タイル素材は100枚以上推奨
• 詳細モードで16レベル設定調整可能
• 処理中でも一時停止・再開可能
• 設定のインポート/エクスポート対応"""
        
        messagebox.showinfo("ヘルプ - Mosaic Art Maker ver3.0", help_text)
    
    def load_config(self):
        """設定ファイルから設定を読み込み"""
        # 基本設定の読み込み
        if CONF_FILE.exists():
            try:
                with open(CONF_FILE, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                
                for key, value in config.items():
                    if key in self.basic_settings:
                        self.basic_settings[key].set(value)
                    elif key == 'source_path' and value:
                        self.source_image_path = try_resolve_or_search(value)
                        self.load_source_image()
                    elif key == 'tile_folder' and value:
                        self.tile_folder_path = try_resolve_or_search(value)
                        self.update_tile_button()
            except Exception as e:
                logger.error("基本設定読み込みエラー: %s", e)

        # 詳細設定（既存） + 旧設定からの移行（face_use_* → face_engine）を吸収（NEW）
        if DETAILED_CONF_FILE.exists():
            try:
                with open(DETAILED_CONF_FILE, 'r', encoding='utf-8') as f:
                    detailed_config = yaml.safe_load(f) or {}
                
                # 旧フラグを face_engine にマッピング（後方互換）
                if 'face_engine' not in detailed_config:
                    if detailed_config.get('face_use_mediapipe'):
                        detailed_config['face_engine'] = 'mediapipe'
                    elif detailed_config.get('face_use_opencv'):
                        detailed_config['face_engine'] = 'opencv'
                    else:
                        detailed_config['face_engine'] = DEFAULT_DETAILED_CONFIG['face_engine']

                for key, value in detailed_config.items():
                    if key in self.advanced_settings:
                        self.advanced_settings[key].set(value)

                self.root.after(100, self.update_weight_display)
            except Exception as e:
                logger.error("詳細設定読み込みエラー: %s", e)

    
    def save_config(self, manual=False):
        """設定をファイルに保存"""
        # 基本設定の保存
        basic_config = {}
        for key, var in self.basic_settings.items():
            basic_config[key] = var.get()

        # 絶対パスはトークン化して保存（CHANGED）
        basic_config['source_path'] = to_portable_path(self.source_image_path)
        basic_config['tile_folder'] = to_portable_path(self.tile_folder_path)

            
        try:
            CONF_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CONF_FILE, 'w', encoding='utf-8') as f:
                yaml.safe_dump(basic_config, f, allow_unicode=True)
        except Exception as e:
            logger.error("基本設定保存エラー: %s", e)
        
        # 詳細設定の保存
        detailed_config = {}
        for key, var in self.advanced_settings.items():
            detailed_config[key] = var.get()
        
        try:
            with open(DETAILED_CONF_FILE, 'w', encoding='utf-8') as f:
                yaml.safe_dump(detailed_config, f, allow_unicode=True)
        except Exception as e:
            logger.error("詳細設定保存エラー: %s", e)
        
        if manual:
            # 手動保存時は通知
            if self.help_label and self.help_label.winfo_exists():
                original_text = self.current_help_text
                self.help_label.configure(text="設定を保存しました")
                self.root.after(3000, lambda: self.help_label.configure(text=original_text))
    
    def manual_save_config(self):
        """手動保存"""
        self.save_config(manual=True)
    
    def reset_to_defaults(self):
        """デフォルト値にリセット"""
        if messagebox.askyesno("設定リセット確認", "すべての設定をデフォルト値にリセットしますか？\n現在の設定は失われます。"):
            # 基本設定をリセット
            for key, var in self.basic_settings.items():
                if key in DEFAULT_CONFIG:
                    var.set(DEFAULT_CONFIG[key])
            
            # 詳細設定をリセット
            for key, var in self.advanced_settings.items():
                if key == 'matching_preset':
                    var.set('basic')  # デフォルトはスタンダード
                elif key in DEFAULT_DETAILED_CONFIG:
                    var.set(DEFAULT_DETAILED_CONFIG[key])
            
            # ファイルパスもリセット
            self.source_image_path = None
            self.tile_folder_path = None
            if hasattr(self, 'source_button'):
                self.source_button.configure(text="画像ファイルを選択してください")
            if hasattr(self, 'tile_button'):
                self.tile_button.configure(text="タイル素材フォルダを選択してください")
            
            # 重み表示を更新
            self.update_weight_display()
            self.update_file_status_labels()
            
            self.save_config(manual=True)
            self.update_statistics()
            self.update_log("設定をデフォルト値にリセットしました")
            
            self.current_help_text = "設定がデフォルト値にリセットされました。新しい設定を始めましょう！"
            self.help_label.configure(text=self.current_help_text)
    
    def select_source_image(self):
        """デザイン画像選択"""
        file_path = filedialog.askopenfilename(
            title="モザイクアートのデザイン画像を選択",
            filetypes=[
                ("画像ファイル", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("JPEG画像", "*.jpg *.jpeg"),
                ("PNG画像", "*.png"),
                ("BMP画像", "*.bmp"),
                ("GIF画像", "*.gif"),
                ("TIFF画像", "*.tiff"),
                ("すべてのファイル", "*.*")
            ],
            initialdir=os.path.expanduser("~/Pictures") if os.path.exists(os.path.expanduser("~/Pictures")) else None
        )
        if file_path:
            self.source_image_path = file_path
            self.load_source_image()
            filename = os.path.basename(file_path)
            if hasattr(self, 'source_button'):
                self.source_button.configure(text=f"選択済み: {filename[:30]}{'...' if len(filename) > 30 else ''}")
            
            # 状況ラベル更新
            self.update_file_status_labels()
            
            self.update_log(f"デザイン画像を読み込みました: {filename}")
            self.current_help_text = f"デザイン画像「{filename}」が選択されました。これがモザイクアートのベースとなります。"
            self.help_label.configure(text=self.current_help_text)
            self.save_config()
    
    def select_tile_folder(self):
        """タイル画像フォルダ選択"""
        folder_path = filedialog.askdirectory(
            title="タイル画像が保存されたフォルダを選択",
            initialdir=os.path.expanduser("~/Pictures") if os.path.exists(os.path.expanduser("~/Pictures")) else None
        )
        if folder_path:
            self.tile_folder_path = folder_path
            self.update_tile_button()
            # 状況ラベル更新
            self.update_file_status_labels()
            self.save_config()
    
    def update_file_status_labels(self):
        """ファイル選択状況ラベルを更新"""
        try:
            if hasattr(self, 'source_status_label') and self.source_status_label:
                if self.source_image_path:
                    filename = os.path.basename(self.source_image_path)
                    self.source_status_label.configure(
                        text=f"デザイン画像: 選択済み {filename}",
                        text_color=COLORS['text_accent']
                    )
                else:
                    self.source_status_label.configure(
                        text="デザイン画像: 未選択",
                        text_color=COLORS['text_secondary']
                    )
            
            if hasattr(self, 'tile_status_label') and self.tile_status_label:
                if self.tile_folder_path:
                    folder_name = os.path.basename(self.tile_folder_path)
                    tile_count = len([p for p in Path(self.tile_folder_path).glob("*") 
                                    if p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".gif")])
                    self.tile_status_label.configure(
                        text=f"タイル素材: 選択済み {folder_name} ({tile_count}枚)",
                        text_color=COLORS['text_accent']
                    )
                else:
                    self.tile_status_label.configure(
                        text="タイル素材: 未選択",
                        text_color=COLORS['text_secondary']
                    )
        except Exception as e:
            logger.debug("ファイル状況ラベル更新エラー: %s", e)
    
    def update_tile_button(self):
        """タイル選択ボタンの表示更新"""
        if self.tile_folder_path:
            tile_count = len([p for p in Path(self.tile_folder_path).glob("*") 
                            if p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".gif")])
            folder_name = os.path.basename(self.tile_folder_path)
            
            if hasattr(self, 'tile_button'):
                display_name = f"{folder_name[:20]}{'...' if len(folder_name) > 20 else ''}"
                self.tile_button.configure(text=f"選択済み: {display_name} ({tile_count}枚)")
            
            self.update_log(f"タイル画像フォルダを設定: {tile_count}枚の画像")
            
            # タイル数に基づく推奨事項
            if tile_count < 50:
                recommendation = "タイル数が少ないです。100枚以上推奨。"
            elif tile_count > 1000:
                recommendation = "多数のタイルで高品質なモザイクが期待できます。"
            else:
                recommendation = "適切なタイル数です。"
            
            self.current_help_text = f"タイル素材フォルダ「{folder_name}」が選択されました。{tile_count}枚の画像が使用されます。{recommendation}"
            self.help_label.configure(text=self.current_help_text)
    
    def load_source_image(self):
        if self.source_image_path:
            try:
                self.source_image = Image.open(self.source_image_path)
                # display_image() ではなく reset_view() を呼んで常に全体表示で開始
                self.reset_view()
                filename = os.path.basename(self.source_image_path)
                if hasattr(self, 'source_button'):
                    self.source_button.configure(text=f"選択済み: {filename}")
                self.update_statistics()
            except Exception as e:
                messagebox.showerror("画像読み込みエラー", f"画像の読み込みに失敗しました:\n{str(e)}\n\n対応形式: JPEG, PNG, BMP, GIF, TIFF")
        
    def refresh_preview(self):
        """エフェクトを適用してプレビューを更新"""
        if self.source_image:
            self.display_image()
    
    def display_image(self):
        """画像をキャンバスに表示（リアルタイムエフェクト付き・自動フィット対応）"""
        if not self.source_image or not hasattr(self, "canvas") or not self.canvas.winfo_exists():
            return

        try:
            # キャンバスサイズ取得
            self.root.update_idletasks()
            canvas_width = max(1, self.canvas.winfo_width())
            canvas_height = max(1, self.canvas.winfo_height())

            # 元画像コピー
            img = self.source_image.copy()

            # 顔トリミング（プレビューにも反映したい場合）
            try:
                if (self.advanced_settings.get('face_crop')
                    and bool(self.advanced_settings['face_crop'].get())):
                    engine = str(self.advanced_settings.get('face_engine', tk.StringVar(value='auto')).get())
                    margin = int(self.advanced_settings.get('face_margin_pct', tk.IntVar(value=20)).get())
                    multi  = str(self.advanced_settings.get('face_multi_mode', tk.StringVar(value='all')).get())
                    target_aspect = (img.width / img.height) if img.height else 1.0
                    img = _face_aware_crop(
                        img, target_aspect,
                        face_engine=engine, multi_mode=multi, margin_pct=margin
                    )
            except Exception as e:
                logger.debug("プレビュー用顔トリミングの適用に失敗: %s", e)

            # エフェクト（明るさ・彩度・ぼかし）
            try:
                bright = float(self.basic_settings['brightness'].get())
                hue = float(self.basic_settings['hue'].get())
                blur_radius = int(self.basic_settings['blur'].get())
            except Exception:
                bright, hue, blur_radius = 1.0, 1.0, 0

            if bright != 1.0:
                img = ImageEnhance.Brightness(img).enhance(bright)
            if hue != 1.0:
                img = ImageEnhance.Color(img).enhance(hue)
            if blur_radius > 0:
                img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            if self.fit_on_next_draw:
            # キャンバスに収まるズーム倍率（拡大はしない）
                fit_zoom_w = canvas_width / img.width if img.width else 1.0
                fit_zoom_h = canvas_height / img.height if img.height else 1.0
                fit_zoom = min(fit_zoom_w, fit_zoom_h)
                if not math.isfinite(fit_zoom) or fit_zoom <= 0:
                    fit_zoom = 1.0
                self.zoom_level = min(1.0, fit_zoom * 0.98)  # 少しだけ余白を持たせる
                self.image_position = [0, 0]
                self.fit_on_next_draw = False

            # ズーム適用
            scaled_w = max(1, int(img.width * self.zoom_level))
            scaled_h = max(1, int(img.height * self.zoom_level))
            resized_image = img.resize((scaled_w, scaled_h), Image.LANCZOS)

            # 表示座標（中央 + ドラッグオフセット）
            x = canvas_width // 2 + self.image_position[0]
            y = canvas_height // 2 + self.image_position[1]

            # 描画
            self.canvas.delete("all")
            self.canvas_image = ImageTk.PhotoImage(resized_image)
            self.canvas_image_id = self.canvas.create_image(x, y, image=self.canvas_image)

            # 画像のキャンバス上の矩形を保存（エリア選択の%換算用）
            left = x - (resized_image.width // 2)
            top  = y - (resized_image.height // 2)
            self._display_geom = {
                'cx': x, 'cy': y,
                'left': left, 'top': top,
                'img_w': resized_image.width, 'img_h': resized_image.height
            }

            # 既存の選択枠があれば再描画
            if self.area_rect_screen:
                x1, y1, x2, y2 = self.area_rect_screen
                self.area_rect_id = self.canvas.create_rectangle(
                    x1, y1, x2, y2, outline="#27AE60", width=2, dash=(4, 2)
                )

            # ズーム表示更新
            if hasattr(self, 'zoom_label'):
                self.zoom_label.configure(text=f"{int(self.zoom_level * 100)}%")

        except Exception as e:
            logger.error("display_image エラー: %s", e)


    def create_face_crop_params(self, parent):
        inner = ctk.CTkFrame(
            parent,
            fg_color=COLORS['warm_white'],
            corner_radius=10,
            border_width=1,
            border_color=COLORS['light_shadow']
        )
        inner.pack(fill="x", padx=10, pady=10)

        # 有効/無効
        chk = ctk.CTkCheckBox(
            inner,
            text="顔認識で自動トリミング(実験的機能)",
            variable=self.advanced_settings.setdefault('face_crop', tk.BooleanVar(value=DEFAULT_DETAILED_CONFIG['face_crop'])),
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS['text_primary']
        )
        chk.pack(anchor="w", padx=12, pady=(12, 6))
        self.add_help_binding(chk, 'face_crop')

        # エンジン選択（NEW）
        ctk.CTkLabel(
            inner, text="顔認識エンジン", font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(anchor="w", padx=12, pady=(6, 4))

        self.advanced_settings.setdefault('face_engine', tk.StringVar(value=DEFAULT_DETAILED_CONFIG['face_engine']))
        radio_row = ctk.CTkFrame(inner, fg_color="transparent"); radio_row.pack(fill="x", padx=8, pady=4)

        for label, val in [("Auto", "auto"), ("MediaPipe", "mediapipe"), ("OpenCV", "opencv")]:
            rb = ctk.CTkRadioButton(
                radio_row, text=label, value=val, variable=self.advanced_settings['face_engine'],
                font=ctk.CTkFont(size=11, weight="bold"), text_color=COLORS['text_primary'],
                radiobutton_width=16, radiobutton_height=16, fg_color=COLORS['primary_blue'],
                hover_color=COLORS['primary_blue_dark']
            )
            rb.pack(side="left", padx=6)
            self.add_help_binding(rb, 'face_engine')

        # マージン / 複数顔
        row = ctk.CTkFrame(inner, fg_color="transparent"); row.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(row, text="マージン（％）:", font=ctk.CTkFont(size=11, weight="bold"),
                    text_color=COLORS['text_primary']).pack(side="left")
        self.advanced_settings.setdefault('face_margin_pct', tk.IntVar(value=DEFAULT_DETAILED_CONFIG['face_margin_pct']))
        ctk.CTkEntry(row, width=80, height=25, textvariable=self.advanced_settings['face_margin_pct']).pack(side="left", padx=10)

        row2 = ctk.CTkFrame(inner, fg_color="transparent"); row2.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(row2, text="複数顔の扱い:", font=ctk.CTkFont(size=11, weight="bold"),
                    text_color=COLORS['text_primary']).pack(side="left")
        self.advanced_settings.setdefault('face_multi_mode', tk.StringVar(value='all'))
        for t, v in [("全てを含める", "all"), ("最大の顔を基準", "largest")]:
            ctk.CTkRadioButton(
                row2, text=t, value=v, variable=self.advanced_settings['face_multi_mode'],
                font=ctk.CTkFont(size=11, weight="bold"), text_color=COLORS['text_primary'],
                radiobutton_width=16, radiobutton_height=16, fg_color=COLORS['primary_blue'],
                hover_color=COLORS['primary_blue_dark']
            ).pack(side="left", padx=6)

        # ※ カスケードXMLの入力/表示は撤去（REMOVE）


    def select_cascade_file(self):
        path = filedialog.askopenfilename(
            title="顔検出用カスケード XML を選択",
            filetypes=[("XML files","*.xml"), ("すべてのファイル","*.*")]
        )
        if path:
            self.advanced_settings['cascade_path'].set(path)
            self.save_config()

    def create_pattern_params(self, parent):
        inner = ctk.CTkFrame(
            parent,
            fg_color=COLORS['warm_white'],
            corner_radius=10,
            border_width=1,
            border_color=COLORS['light_shadow']
        )
        inner.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            inner, text="作成順序パターン", font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(anchor="w", padx=12, pady=(12, 6))

        self.advanced_settings.setdefault('creation_pattern', tk.StringVar(value=DEFAULT_DETAILED_CONFIG['creation_pattern']))
        opt_row = ctk.CTkFrame(inner, fg_color="transparent"); opt_row.pack(fill="x", padx=10, pady=6)

        for label, val in [("ラスタ（標準）","raster"), ("スネーク","snake"), ("ランダム","random"),
                        ("任意エリア優先","area_first"), ("スパイラル","spiral")]:

            rb = ctk.CTkRadioButton(
                opt_row, text=label, value=val, variable=self.advanced_settings['creation_pattern'],
                font=ctk.CTkFont(size=11, weight="bold"), text_color=COLORS['text_primary'],
                radiobutton_width=16, radiobutton_height=16, fg_color=COLORS['primary_blue'],
                hover_color=COLORS['primary_blue_dark']
            )
            rb.pack(side="left", padx=8)
            self.add_help_binding(rb, 'creation_pattern')

        # エリア指定モード（NEW）
        pick_row = ctk.CTkFrame(inner, fg_color="transparent"); pick_row.pack(fill="x", padx=12, pady=(10, 6))
        self.advanced_settings.setdefault('area_picker_enabled', tk.BooleanVar(value=DEFAULT_DETAILED_CONFIG['area_picker_enabled']))
        cb = ctk.CTkCheckBox(
            pick_row, text="プレビューで開始エリアを指定（ドラッグ）",
            variable=self.advanced_settings['area_picker_enabled'],
            font=ctk.CTkFont(size=11, weight="bold"), text_color=COLORS['text_primary'],
            command=self.toggle_area_pick_mode
        )
        cb.pack(side="left")
        self.add_help_binding(cb, 'area_picker_enabled')

        # 任意エリア（％指定）
        area_row = ctk.CTkFrame(inner, fg_color="transparent"); area_row.pack(fill="x", padx=12, pady=(12, 6))
        ctk.CTkLabel(area_row, text="開始エリア（％: x1,y1,x2,y2）",
                    font=ctk.CTkFont(size=11, weight="bold"),
                    text_color=COLORS['text_primary']).pack(side="left")
        self.advanced_settings.setdefault('start_area_pct', tk.StringVar(value=DEFAULT_DETAILED_CONFIG['start_area_pct']))
        area_ent = ctk.CTkEntry(area_row, height=25, width=220, textvariable=self.advanced_settings['start_area_pct'])
        area_ent.pack(side="left", padx=10)
        self.add_help_binding(area_ent, 'start_area_pct')

        ctk.CTkLabel(inner, text="プレビュー上でドラッグすると自動で数値が更新されます。",
                    font=ctk.CTkFont(size=10),
                    text_color=COLORS['text_secondary']).pack(anchor="w", padx=12, pady=(0, 8))
        
    # NEW
    def toggle_area_pick_mode(self):
        self.area_select_active = bool(self.advanced_settings['area_picker_enabled'].get())
        # エリア選択開始時はガイド表示更新
        if self.area_select_active:
            self.current_help_text = "開始エリア指定モード：プレビュー上でドラッグして範囲を選択してください。"
        else:
            self.current_help_text = "開始エリア指定モードを終了しました。"
            # 選択枠を消す
            if self.area_rect_id and self.canvas.winfo_exists():
                self.canvas.delete(self.area_rect_id)
                self.area_rect_id = None
        if self.help_label and self.help_label.winfo_exists():
            self.help_label.configure(text=self.current_help_text)


    
    def on_mouse_wheel(self, event):
        """マウスホイールでズーム"""
        if self.source_image:
            zoom_factor = 1.15 if event.delta > 0 else 0.85
            new_zoom = self.zoom_level * zoom_factor
            
            # ズーム範囲制限
            if 0.2 <= new_zoom <= 8.0:
                self.zoom_level = new_zoom
                self.display_image()
    
    # マウスイベントの上書き/拡張（CHANGED）
    def on_mouse_down(self, event):
        """マウスダウン（ドラッグ開始）"""
        if not self.source_image:
            return
        if self.area_select_active:
            # エリア選択開始
            self.area_rect_screen = (event.x, event.y, event.x, event.y)
            # 既存の枠があれば消す
            if self.area_rect_id:
                self.canvas.delete(self.area_rect_id)
                self.area_rect_id = None
        else:
            # 通常ドラッグ（移動）
            self.is_dragging = True
            self.drag_start = [event.x, event.y]

    def on_mouse_drag(self, event):
        """マウスドラッグ"""
        if not self.source_image:
            return
        if self.area_select_active:
            if not self.area_rect_screen:
                self.area_rect_screen = (event.x, event.y, event.x, event.y)
            x1, y1, _, _ = self.area_rect_screen
            x2, y2 = event.x, event.y
            self.area_rect_screen = (x1, y1, x2, y2)
            # 枠の再描画
            if self.area_rect_id:
                self.canvas.coords(self.area_rect_id, x1, y1, x2, y2)
            else:
                self.area_rect_id = self.canvas.create_rectangle(
                    x1, y1, x2, y2, outline="#27AE60", width=2, dash=(4, 2)
                )
        else:
            if self.is_dragging:
                dx = event.x - self.drag_start[0]
                dy = event.y - self.drag_start[1]
                self.image_position[0] += dx
                self.image_position[1] += dy
                self.drag_start = [event.x, event.y]
                self.display_image()

    def on_mouse_up(self, event):
        """マウスアップ"""
        if self.area_select_active and self.area_rect_screen and self._display_geom:
            # 画面座標 → 画像% に変換
            x1, y1, x2, y2 = self.area_rect_screen
            # 画像のキャンバス上の配置
            left = self._display_geom['left']
            top  = self._display_geom['top']
            iw   = self._display_geom['img_w']
            ih   = self._display_geom['img_h']

            # キャンバス → 画像内正規化（0..1）
            def norm(px, py):
                rx = (px - left) / iw
                ry = (py - top)  / ih
                return max(0, min(1, rx)), max(0, min(1, ry))

            (nx1, ny1) = norm(x1, y1)
            (nx2, ny2) = norm(x2, y2)
            if nx1 > nx2: nx1, nx2 = nx2, nx1
            if ny1 > ny2: ny1, ny2 = ny2, ny1

            pct = f"{int(nx1*100)},{int(ny1*100)},{int(nx2*100)},{int(ny2*100)}"
            self.advanced_settings['start_area_pct'].set(pct)
            self.current_help_text = f"開始エリアを設定しました: {pct}"
            if self.help_label and self.help_label.winfo_exists():
                self.help_label.configure(text=self.current_help_text)

            # 選択は残す（編集可能）。終了したい場合はトグルOFF
            self.area_rect_screen = None
        else:
            self.is_dragging = False

    # 追加：フィット倍率計算
    def _compute_fit_zoom(self) -> float:
        if not self.source_image or not hasattr(self, "canvas") or not self.canvas.winfo_exists():
            return 1.0
        self.root.update_idletasks()
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        iw, ih = self.source_image.width, self.source_image.height
        return min(cw / iw, ch / ih)

    # 差し替え：reset_view は“全体が入る倍率”に
    def reset_view(self):
        self.root.update_idletasks()
        self.zoom_level = self._compute_fit_zoom()
        self.image_position = [0, 0]
        if self.source_image:
            self.display_image()
        self.update_log("プレビュー表示を全体フィットにリセットしました")


    def fit_to_view(self):
        """画像をビューに合わせる"""
        self.zoom_level = 1.0
        self.image_position = [0, 0]
        if self.source_image:
            self.display_image()
        self.update_log("画像をビューに合わせました")
    
    def update_statistics(self):
        """統計情報を更新"""
        if not self.widgets_created or not hasattr(self, 'tile_count_label') or self.tile_count_label is None:
            return
            
        try:
            divisions = self.basic_settings['divisions'].get()
            resize = self.basic_settings['resize'].get()
            
            # None チェックとデフォルト値の設定
            if divisions is None:
                divisions = DEFAULT_CONFIG['divisions']
            if resize is None:
                resize = DEFAULT_CONFIG['resize']
            
            tile_count = divisions * divisions
            
            # ウィジェットの存在確認
            if self.tile_count_label and self.tile_count_label.winfo_exists():
                self.tile_count_label.configure(text=f"タイル数: {tile_count:,}枚")
            if self.image_size_label and self.image_size_label.winfo_exists():
                self.image_size_label.configure(text=f"画像サイズ: {resize:,}px")
            
        except Exception as e:
            logger.error("統計情報更新エラー: %s", e)
    
    def update_log(self, message):
        """ログメッセージを更新"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        logger.info(f"[{timestamp}] {message}")
    
    def update_progress(self, current, total):
        """進行状況を更新"""
        if not self.widgets_created or not hasattr(self, 'progress_bar') or self.progress_bar is None:
            return
            
        try:
            progress = (current / total) * 100
            if self.progress_bar.winfo_exists():
                self.progress_bar.set(progress / 100)
            if hasattr(self, 'progress_label') and self.progress_label and self.progress_label.winfo_exists():
                self.progress_label.configure(text=f"{current:,} / {total:,} ({int(progress)}%)")
            if hasattr(self, 'root') and self.root:
                self.root.update()
        except Exception as e:
            logger.error("進行状況更新エラー: %s", e)
    
    def update_button_states(self):
        """ボタンの状態と色を更新"""
        if not self.widgets_created:
            return
            
        try:
            # RGB生成ボタン
            if hasattr(self, 'rgb_button') and self.rgb_button and self.rgb_button.winfo_exists():
                if self.rgb_state == "running":
                    self.rgb_button.configure(
                        text="RGBリストを作成中... お待ちください", 
                        fg_color="orange", 
                        state="disabled"
                    )
                elif self.rgb_state == "completed":
                    self.rgb_button.configure(
                        text="RGBリスト生成完了", 
                        fg_color=COLORS['text_accent'], 
                        state="normal"
                    )
                else:
                    self.rgb_button.configure(
                        text="RGBリスト生成開始", 
                        fg_color=COLORS['teal_green'], 
                        state="normal"
                    )
            
            # モザイク生成ボタン
            if hasattr(self, 'mosaic_button') and self.mosaic_button and self.mosaic_button.winfo_exists():
                if self.mosaic_state == "running":
                    self.mosaic_button.configure(
                        text="モザイクアート作成中...", 
                        fg_color=COLORS['primary_blue'], 
                        state="normal"
                    )
                elif self.mosaic_state == "paused":
                    self.mosaic_button.configure(
                        text="作成を再開", 
                        fg_color="blue", 
                        state="normal"
                    )
                elif self.mosaic_state == "completed":
                    self.mosaic_button.configure(
                        text="モザイクアート作成完了", 
                        fg_color=COLORS['text_accent'], 
                        state="normal"
                    )
                else:
                    self.mosaic_button.configure(
                        text="モザイクアート作成開始", 
                        fg_color=COLORS['vibrant_pink'], 
                        state="normal"
                    )

            # 一時停止ボタン（修正: 文字色を統一）
            if hasattr(self, 'pause_button') and self.pause_button and self.pause_button.winfo_exists():
                if self.mosaic_state == "running":
                    self.pause_button.configure(
                        text="一時停止",
                        fg_color="red",
                        hover_color="darkred",
                        text_color='#2C3E50',
                        state="normal"
                    )
                elif self.mosaic_state == "paused":
                    self.pause_button.configure(
                        text="再開",
                        fg_color="blue",
                        hover_color="darkblue",
                        text_color='#2C3E50',
                        state="normal"
                    )
                else:
                    self.pause_button.configure(
                        text="一時停止",
                        fg_color="orange",
                        text_color='#2C3E50',
                        state="disabled"
                    )
        except Exception as e:
            logger.error("ボタン状態更新エラー: %s", e)
    
    def get_config(self):
        """現在の設定を辞書で取得（16レベル対応）"""
        config = {}
        for key, var in self.basic_settings.items():
            config[key] = var.get()
        config['source_path'] = self.source_image_path or ""
        config['tile_folder'] = self.tile_folder_path or ""
        
        # 詳細設定から16レベル設定を取得
        if self.ui_mode == "advanced":
            try:
                preset_key = self.advanced_settings['matching_preset'].get()
                if preset_key in MATCHING_PRESETS:
                    preset_data = MATCHING_PRESETS[preset_key]
                    config['matching_levels'] = preset_data['levels']
                    config['level_weights'] = preset_data['weights']
                else:
                    config['matching_levels'] = MATCHING_PRESETS['basic']['levels']
                    config['level_weights'] = MATCHING_PRESETS['basic']['weights']
            except:
                config['matching_levels'] = MATCHING_PRESETS['basic']['levels']
                config['level_weights'] = MATCHING_PRESETS['basic']['weights']
        else:
            # ★変更: 基本モードは basic 固定（=1分割）
            basic_preset = MATCHING_PRESETS['basic']
            config['matching_levels'] = basic_preset['levels']
            config['level_weights'] = basic_preset['weights']

        
        return config
    
    def get_detailed_config(self):
        """詳細設定を辞書で取得"""
        detailed_config = {}
        for key, var in self.advanced_settings.items():
            if key in ['matching_preset']:
                continue  # これらは get_config() で処理済み
            detailed_config[key] = var.get()
        return detailed_config
    
    def generate_rgb_list(self):
        """RGB値リスト生成処理（16レベル対応）"""
        if self.rgb_state == "running":
            return
        
        if not self.tile_folder_path:
            messagebox.showwarning("警告", "タイル画像フォルダを選択してください")
            self.current_help_text = "タイル画像フォルダが選択されていません。先にフォルダを選択してください。"
            self.help_label.configure(text=self.current_help_text)
            return
        
        self.save_config()
        self.rgb_state = "running"
        self.update_button_states()
        
        def process():
            try:
                self.update_log("マルチレベルRGBリストの生成を開始しています...")
                self.current_help_text = "タイル画像のマルチレベルRGB値を分析中です。高品質なマッチングのため時間がかかります。"
                self.help_label.configure(text=self.current_help_text)
                
                save_rgb_values_parallel(self.tile_folder_path, self.update_progress)
                self.update_log("マルチレベルRGBリスト生成が完了しました")
                self.rgb_state = "completed"
                
                self.current_help_text = "マルチレベルRGB値の分析が完了しました！高精度モザイクアートの作成準備が整いました。"
                self.help_label.configure(text=self.current_help_text)
                
            except Exception as e:
                self.update_log(f"RGBリスト生成エラー: {e}")
                self.rgb_state = "idle"
                messagebox.showerror("エラー", f"RGBリスト生成に失敗しました:\n{str(e)}")
                
                self.current_help_text = f"RGBリスト生成でエラーが発生しました: {str(e)}"
                self.help_label.configure(text=self.current_help_text)
            finally:
                self.update_button_states()
        
        threading.Thread(target=process, daemon=True).start()
    
    def toggle_mosaic(self):
        """モザイク作成の開始・再開を制御"""
        if self.mosaic_state == "paused":
            # 再開
            self.pause_flag.clear()
            self.mosaic_state = "running"
            self.update_button_states()
            self.update_statistics()
            self.update_log("モザイク作成を再開しました")
            self.current_help_text = "モザイク作成を再開しています。処理が完了するまでお待ちください。"
            self.help_label.configure(text=self.current_help_text)
            
            config = self.get_config()
            detailed_config = self.get_detailed_config()
            threading.Thread(target=self._mosaic_process, args=(config, detailed_config), daemon=True).start()
            
        else:
            # 新規開始
            if not self.source_image_path or not self.tile_folder_path:
                messagebox.showwarning("必要項目未選択", "デザイン画像とタイル画像フォルダの両方を選択してください")
                self.current_help_text = "デザイン画像とタイル画像フォルダを両方選択してからモザイク作成を開始してください。"
                self.help_label.configure(text=self.current_help_text)
                return
            
            # RGB値ファイルが存在しない場合は自動生成
            rgb_file = OUTPUT_DIR / "rgb_values.json"
            if not rgb_file.exists():
                self.update_log("RGBリストが見つかりません。自動生成を開始します...")
                self.current_help_text = "RGBリストが見つからないため、自動生成を開始しています..."
                self.help_label.configure(text=self.current_help_text)
                self.generate_rgb_list()
                
                # RGB生成完了まで待機
                def wait_and_start():
                    while self.rgb_state == "running":
                        time.sleep(0.1)
                    if self.rgb_state == "completed":
                        self._start_mosaic_creation()
                
                threading.Thread(target=wait_and_start, daemon=True).start()
                return
            
            self._start_mosaic_creation()
    
    def pause_resume_mosaic(self):
        """モザイク作成の一時停止・再開"""
        if self.mosaic_state == "running":
            # 一時停止
            self.pause_flag.set()
            self.mosaic_state = "paused"
            self.update_button_states()
            self.update_log("モザイク作成を一時停止しました")
            self.current_help_text = "モザイク作成が一時停止されました。再開ボタンで続行できます。"
            self.help_label.configure(text=self.current_help_text)
            
        elif self.mosaic_state == "paused":
            # 再開
            self.pause_flag.clear()
            self.mosaic_state = "running"
            self.update_button_states()
            self.update_log("モザイク作成を再開しました")
            self.current_help_text = "モザイク作成を再開しています。"
            self.help_label.configure(text=self.current_help_text)
            
            config = self.get_config()
            detailed_config = self.get_detailed_config()
            threading.Thread(target=self._mosaic_process, args=(config, detailed_config), daemon=True).start()
    
    def _start_mosaic_creation(self):
        """モザイク作成を開始"""
        self.save_config()
        self.pause_flag.clear()
        self.mosaic_state = "running"
        self.update_button_states()
        self.update_statistics()
        
        # マッチングレベルを取得してログに表示
        config = self.get_config()
        matching_levels = config.get('matching_levels', [1, 2, 4])
        level_text = f"{len(matching_levels)}分割"
        self.update_log(f"{level_text}高精度モザイクアート作成を開始しました")
        
        # UIモードに応じたメッセージ
        if self.ui_mode == "advanced":
            self.current_help_text = f"{level_text}詳細設定でモザイクアートを作成中です。進行状況は下部のプログレスバーで確認できます。"
        else:
            self.current_help_text = "基本設定でモザイクアートを作成中です。進行状況は下部のプログレスバーで確認できます。"
        self.help_label.configure(text=self.current_help_text)
        
        detailed_config = self.get_detailed_config()
        threading.Thread(target=self._mosaic_process, args=(config, detailed_config), daemon=True).start()
    
    def _mosaic_process(self, config, detailed_config):
        """モザイク作成処理（16レベル対応）"""
        try:
            # 使用するマッチングレベルを表示
            matching_levels = config.get('matching_levels', [1, 2, 4])
            level_text = f"{len(matching_levels)}分割"
            self.update_log(f"{level_text}高精度モザイクアートを作成中...")
            
            result = create_mosaic(config, detailed_config, self.update_progress, lambda: self.pause_flag.is_set())
            
            if result is None:
                # 一時停止
                self.update_log("モザイク作成が一時停止されました")
                self.current_help_text = "モザイク作成が一時停止されました。再開ボタンで続行してください。"
                self.help_label.configure(text=self.current_help_text)
            else:
                self.update_log(f"{level_text}高精度モザイクアート作成が完了しました！")
                self.mosaic_state = "completed"
                
                output_file, csv_file = result
                
                # 成功メッセージ（UIモードとマッチングレベルに応じて）
                if self.ui_mode == "advanced":
                    success_message = f"{level_text}詳細設定モザイクアートの作成が完了しました！\n\n保存先:\n{output_file}\n\n使用タイル情報:\n{csv_file}\n\n高精度マッチングにより、最高品質のモザイクアートが作成されました。"
                else:
                    success_message = f"モザイクアートの作成が完了しました！\n\n保存先:\n{output_file}\n\n使用タイル情報:\n{csv_file}"
                
                messagebox.showinfo("作成完了", success_message)
                
                self.current_help_text = f"モザイクアート作成完了！ファイルが {output_file.name} として保存されました。"
                self.help_label.configure(text=self.current_help_text)
                
        except Exception as e:
            self.update_log(f"モザイク作成エラー: {e}")
            self.mosaic_state = "idle"
            error_message = f"モザイクアート作成中にエラーが発生しました:\n\n{str(e)}\n\n設定を確認して再度お試しください。"
            messagebox.showerror("作成エラー", error_message)
            
            self.current_help_text = f"モザイク作成でエラーが発生しました: {str(e)}"
            self.help_label.configure(text=self.current_help_text)
        finally:
            self.update_button_states()
            self.update_statistics()

        self.area_select_active = False
        self.area_rect_screen = None   # (x1,y1,x2,y2) in canvas coords
        self.area_rect_id = None       # canvas item id
        self._display_geom = None      # 画像の表示座標情報（display_imageで更新）
    
    def run(self):
        """アプリケーションを実行"""
        try:
            # 初期化完了をチェック
            if self.initialization_complete and self.widgets_created:
                # 初期統計更新
                self.update_statistics()
                self.update_button_states()
                self.update_file_status_labels()
                
                # 初期ヘルプメッセージ
                self.current_help_text = "Mosaic Art Maker へようこそ！まずはデザイン画像とタイル素材を選択しましょう。"
                if self.help_label and self.help_label.winfo_exists():
                    self.help_label.configure(text=self.current_help_text)
                
                # 初期ログメッセージ
                self.update_log("Mosaic Art Maker が正常に起動しました")
                
                # タイトルバーにバージョン情報追加
                self.root.title("Mosaic Art Maker ver3.0")
            else:
                logger.warning("初期化が不完全な状態でrun()が呼ばれました")
                
        except Exception as e:
            logger.error("初期化エラー: %s", e)
            try:
                logger.warning(f"初期化中にエラーが発生しましたが、起動を続行します: {e}")
            except:
                print(f"ログ出力に失敗: {e}")
        
        # ウィンドウを表示
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except Exception as e:
            print(f"アプリケーション実行エラー: {e}")
            print("設定ファイルを削除して再度お試しください。")
    
    def on_closing(self):
        """アプリケーション終了時の処理"""
        try:
            # 進行中の処理があれば停止
            if self.mosaic_state == "running":
                if messagebox.askyesno("確認", "モザイク作成が進行中です。終了しますか？"):
                    self.pause_flag.set()
                else:
                    return
            
            # 設定を保存
            self.save_config()
            self.update_log("アプリケーションを終了します")
            
        except Exception as e:
            logger.error("終了処理エラー: %s", e)
        finally:
            self.root.destroy()

def main():
    """メイン関数"""
    if len(sys.argv) > 1 and sys.argv[1] == "rgb":
        # コマンドライン: RGBリスト生成のみ
        try:
            folder = input("タイル画像フォルダのパス: ")
            print("16レベルRGB値を生成中...")
            save_rgb_values_parallel(folder)
            print("完了！")
        except Exception as e:
            print(f"RGBリスト生成エラー: {e}")
    elif len(sys.argv) > 1 and sys.argv[1] == "mosaic":
        # コマンドライン: モザイク作成のみ
        try:
            config = DEFAULT_CONFIG.copy()
            detailed_config = DEFAULT_DETAILED_CONFIG.copy()
            config['source_path'] = input("デザイン画像のパス: ")
            config['tile_folder'] = input("タイル画像フォルダのパス: ")
            print("16レベル高精度モザイクを作成中...")
            create_mosaic(config, detailed_config)
            print("完了！")
        except Exception as e:
            print(f"モザイク作成エラー: {e}")
    else:
        # GUI モード
        try:
            print("=" * 60)
            print("Mosaic Artr Maker ver3.0")
            print("=" * 60)
            print("新機能:")
            print("  • 16レベル高精度マッチング")
            print("  • 基本設定/詳細設定の段階表示UI")
            print("  • 一時停止・再開機能")
            print("  • リアルタイムプレビュー機能")
            print("  • キーボードショートカット対応")
            print("  • 設定インポート/エクスポート")
            print("=" * 60)
            print("起動中...")
            
            app = MosaicMaker()
            app.run()
        except ImportError as e:
            print(f"\nエラー: 必要なライブラリがインストールされていません: {e}")
            print("\n以下のコマンドでインストールしてください:")
            print("pip install customtkinter Pillow numpy pyyaml psutil")
            print("\nまたは requirements.txt がある場合:")
            print("pip install -r requirements.txt")
        except Exception as e:
            print(f"\n予期しないエラーが発生しました: {e}")
            print("\n問題が解決しない場合は、以下を確認してください:")
            print("1. Pythonのバージョンが3.8以上であること")
            print("2. 必要なライブラリがすべてインストールされていること")
            print("3. CustomTkinterのバージョンが最新であること")
            print("4. 設定ファイルを削除して再度お試しください")
            print(f"\n設定ファイルの場所: {SETTING_DIR}")

if __name__ == "__main__":
    try:
        mp.freeze_support()
        main()
    except KeyboardInterrupt:
        print("\nアプリケーションが中断されました")
    except Exception as e:
        print(f"\n致命的なエラー: {e}")
        print("アプリケーションを終了します。")
