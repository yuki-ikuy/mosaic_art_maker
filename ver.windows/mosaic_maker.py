
from __future__ import annotations
import sys, os, shutil, threading, queue, re, yaml, ast, csv, logging
import multiprocessing as mp, warnings, time
from collections import deque
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
import uuid

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

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
SETTING_DIR = BASE_DIR / "setting"; SETTING_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR  = BASE_DIR / "output";  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONF_FILE   = SETTING_DIR / "default_config.yaml"

DEFAULT: Dict[str, Any] = dict(
    original_image_path    = "",
    tile_images_folder     = "",
    num_title              = 100,
    resize_base_px         = 10000,
    max_tile_usage         = 2,
    link_vertical_horizontal=True,
    repeat_limit           = 2,
    vertical_limit         = 2,
    horizontal_limit       = 2,
    brightness_adjust_on   = True,
    brightness_factor      = 1.0,
    color_adjust_on        = False,
    color_factor           = 1.0,
    gaussian_blur_on       = False,
    blur_ksize             = 3,
    dpi                    = 300,
    jpeg_quality           = 50,
    sub_blocks             = 10,
    rgb_values_file        = "output/rgb_values.txt",
)
INT_KEYS   = {"num_title","resize_base_px","max_tile_usage","sub_blocks","blur_ksize",
              "jpeg_quality","dpi","repeat_limit","vertical_limit","horizontal_limit"}
FLOAT_KEYS = {"brightness_factor","color_factor"}

if CONF_FILE.exists():
    try:
        DEFAULT.update(yaml.safe_load(CONF_FILE.read_text(encoding="utf-8")) or {})
    except Exception as e:
        logger.error("設定ファイル読み込みエラー: %s", e)
else:
    CONF_FILE.write_text(yaml.safe_dump(DEFAULT, allow_unicode=True), encoding="utf-8")

def save_config(cfg: Dict[str, Any]):
    """GUI から呼ばれる設定保存"""
    try:
        with open(CONF_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)
    except Exception as e:
        logger.error("設定保存失敗: %s", e)

# ───────── RGBリスト生成 ─────────
def calculate_average_colors(path: str):
    p = Path(path)
    try:
        img = Image.open(p).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        logger.warning("画像読み込み失敗: %s → %s", p.name, e)
        return None
    w, h = img.size
    sw, sh = w//4, h//4
    avgs=[]
    for j in range(4):
        for i in range(4):
            crop = img.crop((i*sw, j*sh, (i+1)*sw, (j+1)*sh))
            arr  = np.asarray(crop)
            avgs.append(tuple(int(x) for x in arr.mean(axis=(0,1)))) if arr.size else avgs.append((0,0,0))
    return p.name, avgs

def _worker_rgb(img_path, res_list, prog_q, skip_q):
    r = calculate_average_colors(img_path)
    if r: res_list.append(r)
    else: skip_q.put(Path(img_path).name)
    prog_q.put(1)

def save_rgb_values_parallel(cfg=None, progress=None):
    from multiprocessing import Pool, Manager
    cfg = cfg or DEFAULT
    tile_dir = BASE_DIR / cfg["tile_images_folder"]
    out_path = BASE_DIR / cfg["rgb_values_file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [str(p) for p in tile_dir.glob("*") if p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".gif")]
    if len(imgs) < 2:
        raise RuntimeError("タイル画像が2枚未満です")
    m = Manager(); res_list, prog_q, skip_q = m.list(), m.Queue(), m.Queue()
    total = len(imgs)
    def cb(done):
        if progress: progress(done/total, done, total)
    with Pool(processes=os.cpu_count() or 4) as pool:
        for p in imgs: pool.apply_async(_worker_rgb, args=(p, res_list, prog_q, skip_q))
        done = 0
        while done < total:
            prog_q.get(); done += 1; cb(done)
        pool.close(); pool.join()
    skipped = []
    while not skip_q.empty(): skipped.append(skip_q.get())
    if skipped:
        logger.warning("読み込み失敗画像: %s", ", ".join(skipped))
    with open(out_path, "w", encoding="utf-8") as f:
        for fn, avg in res_list:
            f.write(f"{fn}: {avg}\n")
    logger.info("RGB値保存 → %s", out_path)
    cfg["rgb_values_file"] = str(out_path)
    save_config(cfg)
    return out_path

# ───────── モザイク生成─────────
def _resize_keep(img: Image.Image, base: int):
    w, h = img.size
    return img.resize((base, int(h * base / w)), Image.LANCZOS) if w > h \
        else img.resize((int(w * base / h), base), Image.LANCZOS)

def _load_rgb_file(path: Path) -> dict[str, tuple[int, int, int]]:
    table: dict[str, tuple[int, int, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line: continue
        fn, val = line.split(":", 1)
        table[fn.strip()] = tuple(np.mean(ast.literal_eval(val.strip()), axis=0).astype(int))
    return table

def _find_closest(target, rgb_tbl, used, recent, x, y, mmap, n, limit):
    best, best_d = None, float("inf")
    for name, col in rgb_tbl.items():
        if used[name] >= limit or name in recent:
            continue
        if (x and mmap[y * n + x - 1] == name) or (y and mmap[(y - 1) * n + x] == name):
            continue
        d = np.linalg.norm(target - np.asarray(col))
        if d < best_d:
            best, best_d = name, d
    return best

def _adj_brightness(tile: Image.Image, tgt_rgb: np.ndarray) -> Image.Image:
    mean_tile, mean_tgt = np.mean(tile), np.mean(tgt_rgb)
    return ImageEnhance.Brightness(tile).enhance(mean_tgt / mean_tile) if mean_tile > 1e-5 else tile


def _block_worker_return(args):
    x, y, block_arr, rgb_tbl, bw, bh, n, limit, tile_dir = args
    target_rgb = block_arr.mean(axis=(0, 1))
    name = None
    best_d = float("inf")
    for k, v in rgb_tbl.items():
        d = np.linalg.norm(target_rgb - np.asarray(v))
        if d < best_d:
            best_d = d
            name = k
    return (x, y, name)

def create_mosaic(cfg: Dict[str, Any] | None = None, progress=None):
    import gc
    cfg = cfg or DEFAULT
    MAX_TILES = 200
    n = int(cfg.get("num_title", 100))
    if n > MAX_TILES:
        print(f"ERROR: 分割数が多すぎます（上限{MAX_TILES}）。設定を見直してください。")
        return

    base = Path(cfg["original_image_path"])
    if not base.is_absolute():
        base = BASE_DIR / base
    if not base.is_file():
        print("ERROR: 元画像が見つかりません"); return
    rgb_txt = BASE_DIR / cfg["rgb_values_file"]
    if not rgb_txt.is_file():
        print("ERROR: rgb_values.txt が見つかりません"); return
    tile_dir = BASE_DIR / cfg["tile_images_folder"]
    if not tile_dir.is_dir():
        print("ERROR: タイルフォルダが見つかりません"); return

    rgb_tbl = _load_rgb_file(rgb_txt)
    img = _resize_keep(Image.open(base).convert("RGB"), cfg["resize_base_px"])
    w, h = img.size
    bw, bh = w // n, h // n
    total = n * n

    used = {k: 0 for k in rgb_tbl}
    mmap = [None] * total
    mosaic = Image.new("RGB", (w, h))

    for i, (y, x) in enumerate(((y, x) for y in range(n) for x in range(n))):
        # 端のタイルは画像の右端・下端で切り出す
        left   = x * bw
        upper  = y * bh
        right  = (x + 1) * bw if x < n - 1 else w
        lower  = (y + 1) * bh if y < n - 1 else h
        block = img.crop((left, upper, right, lower))
        block_arr = np.asarray(block)
        target_rgb = block_arr.mean(axis=(0, 1))
        name = None
        best_d = float("inf")
        for k, v in rgb_tbl.items():
            if used[k] >= cfg["max_tile_usage"]:
                continue
            d = np.linalg.norm(target_rgb - np.asarray(v))
            if d < best_d:
                best_d = d
                name = k
        mmap[y * n + x] = name
        if name:
            used[name] += 1
            tile_path = tile_dir / name
            if tile_path.exists():
                # タイルをそのブロックサイズに合わせてリサイズ
                tile = Image.open(tile_path).convert("RGB").resize((right-left, lower-upper))
                mosaic.paste(tile, (left, upper))
        if progress:
            progress((i+1)/total, i+1, total)
        del block_arr; gc.collect()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_png = OUTPUT_DIR / f"mosaic_{ts}.png"
    out_csv = OUTPUT_DIR / f"used_{ts}.csv"

    try:
        mosaic.save(out_png)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([["filename", "used_count"], *sorted(used.items())])
    except Exception as e:
        logger.error("保存時エラー: %s", e)
        print(f"ERROR: 保存失敗 {e}")
        return

    logger.info("モザイク保存 → %s", out_png.resolve())
    logger.info("CSV保存 → %s", out_csv.resolve())
    print(f"\nモザイク画像を保存しました: {out_png.resolve()}")
    print(f"CSVも保存しました: {out_csv.resolve()}")
    print("DONE", flush=True)
    return out_png, out_csv



# ────────── GUI ──────────
def _run_gui():
    import tkinter as tk, customtkinter as ctk
    from customtkinter import CTkImage
    from tkinter import filedialog, messagebox

    class MosaicApp(ctk.CTk):
        def __init__(self):
            super().__init__()
            self.title("Mosaic-Maker")
            self.geometry("1400x900"); self.minsize(1100, 750)
            bar = ctk.CTkFrame(self, width=120); bar.pack(side="left", fill="y")
            ctk.CTkLabel(bar, text="メニュー", font=("", 16, "bold")).pack(pady=20)
            ctk.CTkButton(bar, text="設定", command=self.show_cfg).pack(pady=10)
            ctk.CTkButton(bar, text="終了", command=self.destroy).pack(side="bottom", pady=20)
            self.container = ctk.CTkFrame(self); self.container.pack(fill="both", expand=True)
            self.cfg_page  = ConfigPage(self.container); self.show_cfg()
        def show_cfg(self): self.cfg_page.pack(fill="both", expand=True)

    class ConfigPage(ctk.CTkFrame):
        PREVIEW_BOX = (480, 480)
        def __init__(self, master):
            super().__init__(master)
            self.grid_columnconfigure(0, weight=1)
            self.grid_columnconfigure(1, weight=0)
            self.grid_rowconfigure(0, weight=1)
            self.vars  = {k: tk.StringVar(value=str(v)) for k, v in DEFAULT.items()}
            bool_keys  = ["link_vertical_horizontal","brightness_adjust_on","color_adjust_on","gaussian_blur_on"]
            self.bools = {k: tk.IntVar(value=1 if DEFAULT[k] else 0) for k in bool_keys}
            form = ctk.CTkScrollableFrame(self)
            form.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
            def add_row(lbl, widget):
                r = form.grid_size()[1]
                ctk.CTkLabel(form, text=lbl).grid(row=r, column=0, sticky="w", padx=4, pady=2)
                widget.grid(row=r, column=1, sticky="w", padx=4, pady=2)
            add_row("元画像", ctk.CTkButton(form, text="選択", command=lambda: self.pick("original_image_path", True)))
            add_row("タイルフォルダ", ctk.CTkButton(form, text="選択", command=lambda: self.pick("tile_images_folder", False)))
            def num_entry(key, flt=False):
                vcmd = (self.register(lambda s, f=flt: re.fullmatch(
                        r"^\d*"+(r"(\.\d*)?" if f else "")+r"$", s) is not None), "%P")
                return ctk.CTkEntry(form, width=90, textvariable=self.vars[key],
                                    validate="key", validatecommand=vcmd)
            for k,l,fl in [
                ("num_title","分割数",False),("resize_base_px","リサイズ基準",False),
                ("max_tile_usage","同一画像最大使用数",False),
                ("repeat_limit","連続使用制限",False),("vertical_limit","縦方向制限",False),
                ("horizontal_limit","横方向制限",False),
                ("brightness_factor","明るさ係数",True),
                ("color_factor","色合い係数",True),
                ("blur_ksize","Blur ksize",False),
                ("dpi","DPI",False)
            ]: add_row(l, num_entry(k, fl))
            for k,l in [("link_vertical_horizontal","縦横連携"),
                        ("brightness_adjust_on","明るさ補正 ON"),
                        ("color_adjust_on","色合い補正 ON"),
                        ("gaussian_blur_on","ガウシアンぼかし ON")]:
                add_row("", ctk.CTkCheckBox(form, text=l, variable=self.bools[k]))
            ctk.CTkButton(form, text="設定を保存", command=self.save_cfg).grid(pady=8, columnspan=2)
            ctk.CTkButton(form, text="RGBリスト生成", command=self.start_rgb).grid(pady=4, columnspan=2)
            ctk.CTkButton(form, text="モザイク生成開始", command=self.start_mosaic).grid(pady=4, columnspan=2)
            self.preview_box = ctk.CTkFrame(self, width=self.PREVIEW_BOX[0], height=self.PREVIEW_BOX[1])
            self.preview_box.grid_propagate(False)
            self.preview_box.grid(row=0, column=1, padx=(5, 10), pady=10)
            self.preview_lbl = ctk.CTkLabel(self.preview_box, text="")
            self.preview_lbl.pack(expand=True)
            self.zoom = tk.DoubleVar(value=1.0)
            ctk.CTkSlider(self, from_=0.5, to=3, number_of_steps=25, variable=self.zoom,
                          command=lambda *_: self.refresh()).grid(row=1, column=1, sticky="ew", padx=(5, 10))
            ctk.CTkLabel(self, text="ズーム").grid(row=1, column=1, sticky="w", padx=(5, 50))
            self.prog = ctk.CTkProgressBar(self)
            self.prog.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10)
            self.log = tk.StringVar()
            ctk.CTkLabel(self, textvariable=self.log).grid(row=3, column=0, columnspan=2,
                                                          sticky="w", padx=12, pady=(0, 8))
            for k in ["original_image_path","brightness_factor","color_factor","blur_ksize"]:
                self.vars[k].trace_add("write", lambda *_: self.refresh())
            for b in self.bools.values():
                b.trace_add("write", lambda *_: self.refresh())
            self.refresh()
        def pick(self, key, file):
            self.vars[key].set(filedialog.askopenfilename() if file else filedialog.askdirectory())
        def cfg(self):
            cfg = {k: (v.get() if k not in self.bools else bool(self.bools[k].get()))
                   for k, v in self.vars.items()}
            for k,b in self.bools.items(): cfg[k]=bool(b.get())
            for k in INT_KEYS:   cfg[k] = int(cfg[k])
            for k in FLOAT_KEYS: cfg[k] = float(cfg[k])
            return cfg
        def save_cfg(self):
            save_config(self.cfg())
            messagebox.showinfo("保存", "設定を保存しました")
        def refresh(self):
            p = Path(self.vars["original_image_path"].get())
            if not p.is_file():
                self.preview_lbl.configure(text="元画像未選択", image=None)
                return
            try:
                img = Image.open(p).convert("RGB")
                if self.bools["brightness_adjust_on"].get():
                    img = ImageEnhance.Brightness(img).enhance(float(self.vars["brightness_factor"].get() or 1))
                if self.bools["color_adjust_on"].get():
                    img = ImageEnhance.Color(img).enhance(float(self.vars["color_factor"].get() or 1))
                if self.bools["gaussian_blur_on"].get():
                    k = int(self.vars["blur_ksize"].get() or 1) | 1
                    img = img.filter(ImageFilter.GaussianBlur(radius=k))
                bw,bh = self.PREVIEW_BOX
                w,h = img.size
                r = min(bw/w, bh/h) * self.zoom.get()
                img = img.resize((int(w*r), int(h*r)))
                cimg = CTkImage(light_image=img, size=img.size)
                self.preview_lbl.configure(image=cimg, text=""); self.preview_lbl.image = cimg
            except Exception as e:
                self.preview_lbl.configure(text=f"プレビュー失敗:{e}", image=None)
        def _thread_run(self, target):
            def cb(rate, done, total): self.prog.set(rate); self.log.set(f"({total}枚中{done}枚) {rate*100:.1f}%")
            try: target(cb)
            except Exception as e: self.log.set(f"ERROR:{e}")
            else: self.prog.set(1); self.log.set("完了！")
        def start_rgb(self):
            cfg=self.cfg(); threading.Thread(target=self._thread_run,args=(lambda cb:save_rgb_values_parallel(cfg,cb),),daemon=True).start()
        def start_mosaic(self):
            cfg=self.cfg(); threading.Thread(target=self._thread_run,args=(lambda cb:create_mosaic(cfg,cb),),daemon=True).start()

    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    MosaicApp().mainloop()

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "rgb":
        save_rgb_values_parallel(DEFAULT, lambda *_:None)
    elif len(sys.argv) > 1 and sys.argv[1] == "mosaic":
        create_mosaic(DEFAULT, lambda *_:None)
    else:
        _run_gui()

if __name__ == "__main__":
    mp.freeze_support()  
    main()
