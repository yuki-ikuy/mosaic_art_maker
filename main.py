
import sys, subprocess, queue, threading, yaml, re
from pathlib import Path
import tkinter as tk
import customtkinter as ctk
from customtkinter import CTkImage
from tkinter import filedialog, messagebox
from PIL import Image, ImageEnhance, ImageFilter

# ── パス & 設定ファイル ──────────────────────────
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

SETTING_DIR = BASE_DIR / "setting"; SETTING_DIR.mkdir(exist_ok=True)
OUTPUT_DIR  = BASE_DIR / "output";  OUTPUT_DIR.mkdir(exist_ok=True)
CONF_FILE   = SETTING_DIR / "default_config.yaml"

DEFAULT = dict(
    original_image_path  ="",  tile_images_folder ="",
    num_title=100, resize_base_px=10000, max_tile_usage=2,
    link_vertical_horizontal=True, repeat_limit=2,
    vertical_limit=2, horizontal_limit=2,
    brightness_adjust_on=True, brightness_factor=1.0,
    color_adjust_on=False,     color_factor=1.0,
    gaussian_blur_on=False,    blur_ksize=3,
    dpi=300, jpeg_quality=50,  sub_blocks=10
)
if CONF_FILE.exists():
    DEFAULT.update(yaml.safe_load(CONF_FILE.read_text(encoding="utf-8")) or {})
else:
    CONF_FILE.write_text(yaml.safe_dump(DEFAULT, allow_unicode=True), encoding="utf-8")

# ── GUI アプリ ───────────────────────────────────
class MosaicApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Mosaic‑Maker"); self.geometry("1400x900"); self.minsize(1100,750)

        bar = ctk.CTkFrame(self, width=120); bar.pack(side="left", fill="y")
        ctk.CTkLabel(bar, text="メニュー", font=("",16,"bold")).pack(pady=20)
        ctk.CTkButton(bar, text="設定", command=self.show_cfg).pack(pady=10)
        ctk.CTkButton(bar, text="終了", command=self.destroy).pack(side="bottom", pady=20)

        self.container = ctk.CTkFrame(self); self.container.pack(fill="both", expand=True)
        self.cfg_page  = ConfigPage(self.container)
        self.show_cfg()

    def show_cfg(self): self.cfg_page.pack(fill="both", expand=True)

# ── 設定ページ ───────────────────────────────────
class ConfigPage(ctk.CTkFrame):
    PREVIEW_BOX = (480,480)

    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1); self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        self.vars  = {k: tk.StringVar(value=str(v)) for k,v in DEFAULT.items()}
        bool_keys  = ["link_vertical_horizontal","brightness_adjust_on",
                      "color_adjust_on","gaussian_blur_on"]
        self.bools = {k: tk.IntVar(value=1 if DEFAULT[k] else 0) for k in bool_keys}

        form = ctk.CTkScrollableFrame(self); form.grid(row=0,column=0,sticky="nsew",padx=(10,5),pady=10)
        def add_row(lbl, wid):
            r=form.grid_size()[1]; ctk.CTkLabel(form,text=lbl).grid(row=r,column=0,sticky="w",padx=4,pady=2)
            wid.grid(row=r,column=1,sticky="w",padx=4,pady=2)

        add_row("元画像", ctk.CTkButton(form,text="選択", command=lambda:self.pick("original_image_path",True)))
        add_row("タイルフォルダ", ctk.CTkButton(form,text="選択", command=lambda:self.pick("tile_images_folder",False)))

        def num_entry(key, is_float=False):
            vcmd = (self.register(lambda s,flt=is_float: re.fullmatch(r'^\d*' + (r'(\.\d*)?' if flt else '') + r'$',s) is not None), "%P")
            e = ctk.CTkEntry(form, width=90, textvariable=self.vars[key], validate="key", validatecommand=vcmd)
            return e
        for k,l,is_f in [
            ("num_title","分割数",False), ("resize_base_px","リサイズ基準",False),
            ("max_tile_usage","同一画像最大使用数",False), ("repeat_limit","連続使用制限",False),
            ("vertical_limit","縦方向制限",False), ("horizontal_limit","横方向制限",False),
            ("brightness_factor","明るさ係数",True), ("color_factor","色合い係数",True),
            ("blur_ksize","Blur ksize(奇数)",False), ("dpi","DPI",False),
            ("jpeg_quality","JPEG品質",False)
        ]: add_row(l, num_entry(k,is_f))

        for k,l in [("link_vertical_horizontal","縦横連携"),
                    ("brightness_adjust_on","明るさ補正 ON"),
                    ("color_adjust_on","色合い補正 ON"),
                    ("gaussian_blur_on","ガウシアンぼかし ON")]:
            add_row("", ctk.CTkCheckBox(form,text=l,variable=self.bools[k]))

        ctk.CTkButton(form,text="設定を保存",command=self.save_cfg).grid(pady=8,columnspan=2)
        ctk.CTkButton(form,text="RGBリスト生成",command=self.start_rgb_gen).grid(pady=4,columnspan=2)
        ctk.CTkButton(form,text="モザイク生成開始",command=self.start_job).grid(pady=4,columnspan=2)

        self.preview_box=ctk.CTkFrame(self,width=self.PREVIEW_BOX[0],height=self.PREVIEW_BOX[1])
        self.preview_box.grid_propagate(False)
        self.preview_box.grid(row=0,column=1,padx=(5,10),pady=10)
        self.preview_lbl=ctk.CTkLabel(self.preview_box,text=""); self.preview_lbl.pack(expand=True)

        self.zoom=tk.DoubleVar(value=1.0)
        ctk.CTkSlider(self,from_=0.5,to=3,number_of_steps=25,variable=self.zoom,
                      command=lambda *_:self.refresh()).grid(row=1,column=1,sticky="ew",padx=(5,10))
        ctk.CTkLabel(self, text="ズーム").grid(row=1, column=1, sticky="w", padx=(5, 50))


        self.prog=ctk.CTkProgressBar(self); self.prog.grid(row=2,column=0,columnspan=2,sticky="ew",padx=10)
        self.log=tk.StringVar(); ctk.CTkLabel(self,textvariable=self.log)\
            .grid(row=3,column=0,columnspan=2,sticky="w",padx=12,pady=(0,8))

        for k in ["original_image_path","brightness_factor","color_factor","blur_ksize"]:
            self.vars[k].trace_add("write", lambda *_: self.refresh())
        for b in self.bools.values(): b.trace_add("write", lambda *_: self.refresh())

        self.refresh()

    def pick(self,key, file): 
        path = filedialog.askopenfilename() if file else filedialog.askdirectory()
        if path: self.vars[key].set(path)

    def cfg(self):
        cfg={k:(v.get() if k not in self.bools else bool(self.bools[k].get()))
             for k,v in self.vars.items()}
        for k,b in self.bools.items(): cfg[k]=bool(b.get()); return cfg

    def save_cfg(self):
        CONF_FILE.write_text(yaml.safe_dump(self.cfg(),allow_unicode=True),"utf-8")
        messagebox.showinfo("保存","設定を保存しました")

    def refresh(self):
        path = Path(self.vars["original_image_path"].get())
        if not path.is_file():
            self.preview_lbl.configure(text="元画像未選択",image=None); return
        img = Image.open(path).convert("RGB")
        if self.bools["brightness_adjust_on"].get():
            img = ImageEnhance.Brightness(img).enhance(float(self.vars["brightness_factor"].get() or 1))
        if self.bools["color_adjust_on"].get():
            img = ImageEnhance.Color(img).enhance(float(self.vars["color_factor"].get() or 1))
        if self.bools["gaussian_blur_on"].get():
            k=int(self.vars["blur_ksize"].get() or 1)|1
            img=img.filter(ImageFilter.GaussianBlur(radius=k))
        bw,bh=self.PREVIEW_BOX; w,h=img.size
        ratio=min(bw/w,bh/h)*self.zoom.get(); img=img.resize((int(w*ratio),int(h*ratio)))
        cimg=CTkImage(light_image=img,size=img.size)
        self.preview_lbl.configure(image=cimg,text=""); self.preview_lbl.image=cimg

    def start_job(self):
        if not Path(self.vars["original_image_path"].get()).is_file():
            messagebox.showerror("エラー","元画像を選択してください"); return
        if not Path(self.vars["tile_images_folder"].get()).is_dir():
            messagebox.showerror("エラー","タイルフォルダを選択してください"); return
        self.save_cfg(); self.prog.set(0); self.log.set("モザイク生成中…")
        self.q=queue.Queue(); threading.Thread(target=self._worker,args=("mosaic_generator.py",),daemon=True).start()
        self.after(100,self._poll)

    def start_rgb_gen(self):
        if not Path(self.vars["tile_images_folder"].get()).is_dir():
            messagebox.showerror("エラー","タイルフォルダを選択してください"); return
        self.save_cfg(); self.prog.set(0); self.log.set("RGBリスト生成中…")
        self.q=queue.Queue(); threading.Thread(target=self._worker,args=("rgb_generator.py",),daemon=True).start()
        self.after(100,self._poll)

    def _worker(self, exe_name):
        exe_path = Path(__file__).parent / exe_name
        if not exe_path.exists():
            self.q.put(f"エラー: {exe_name} が見つかりません")
            self.q.put("DONE")
            return

        try:
            with subprocess.Popen(
                [sys.executable, str(exe_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            ) as p:
                for line in p.stdout:
                    self.q.put(line.rstrip())
        except Exception as e:
            self.q.put(f"実行中にエラーが発生しました: {e}")
        finally:
            self.q.put("DONE")

    def _poll(self):
        try:
            while True:
                line=self.q.get_nowait()
                if line.startswith("PROG:"): self.prog.set(float(line.split(":")[1]))
                elif line=="DONE": self.prog.set(1); self.log.set("完了！"); return
                else: self.log.set(line)
        except queue.Empty: pass
        self.after(100,self._poll)

# ── 起動 ───────────────────────────────────────
if __name__=="__main__":
    ctk.set_appearance_mode("System"); ctk.set_default_color_theme("blue")
    MosaicApp().mainloop()


'''

pyinstaller main.py --noconfirm \
--add-data "setting:setting" \
--add-data "output:output" \
--add-data "mosaic_generator.py:." \
--add-data "rgb_generator.py:."

'''

'''''

import sys, queue, threading, yaml, re
from pathlib import Path
import tkinter as tk
import customtkinter as ctk
from customtkinter import CTkImage
from tkinter import filedialog, messagebox
from PIL import Image, ImageEnhance, ImageFilter

# 新たに追加
from rgb_generator import save_rgb_values_parallel
from mosaic_generator import generate_mosaic_image

# ── パス & 設定ファイル ──────────────────────────
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

SETTING_DIR = BASE_DIR / "setting"; SETTING_DIR.mkdir(exist_ok=True)
OUTPUT_DIR  = BASE_DIR / "output";  OUTPUT_DIR.mkdir(exist_ok=True)
CONF_FILE   = SETTING_DIR / "default_config.yaml"

DEFAULT = dict(
    original_image_path  ="",  tile_images_folder ="",
    num_title=100, resize_base_px=10000, max_tile_usage=2,
    link_vertical_horizontal=True, repeat_limit=2,
    vertical_limit=2, horizontal_limit=2,
    brightness_adjust_on=True, brightness_factor=1.0,
    color_adjust_on=False,     color_factor=1.0,
    gaussian_blur_on=False,    blur_ksize=3,
    dpi=300, jpeg_quality=50,  sub_blocks=10
)
if CONF_FILE.exists():
    DEFAULT.update(yaml.safe_load(CONF_FILE.read_text(encoding="utf-8")) or {})
else:
    CONF_FILE.write_text(yaml.safe_dump(DEFAULT, allow_unicode=True), encoding="utf-8")

# ── GUI アプリ ───────────────────────────────────
class MosaicApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Mosaic‑Maker"); self.geometry("1400x900"); self.minsize(1100,750)

        bar = ctk.CTkFrame(self, width=120); bar.pack(side="left", fill="y")
        ctk.CTkLabel(bar, text="メニュー", font=("",16,"bold")).pack(pady=20)
        ctk.CTkButton(bar, text="設定", command=self.show_cfg).pack(pady=10)
        ctk.CTkButton(bar, text="終了", command=self.destroy).pack(side="bottom", pady=20)

        self.container = ctk.CTkFrame(self); self.container.pack(fill="both", expand=True)
        self.cfg_page  = ConfigPage(self.container)
        self.show_cfg()

    def show_cfg(self): self.cfg_page.pack(fill="both", expand=True)

# ── 設定ページ ───────────────────────────────────
class ConfigPage(ctk.CTkFrame):
    PREVIEW_BOX = (480,480)

    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1); self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        self.vars  = {k: tk.StringVar(value=str(v)) for k,v in DEFAULT.items()}
        bool_keys  = ["link_vertical_horizontal","brightness_adjust_on",
                      "color_adjust_on","gaussian_blur_on"]
        self.bools = {k: tk.IntVar(value=1 if DEFAULT[k] else 0) for k in bool_keys}

        form = ctk.CTkScrollableFrame(self); form.grid(row=0,column=0,sticky="nsew",padx=(10,5),pady=10)
        def add_row(lbl, wid):
            r=form.grid_size()[1]; ctk.CTkLabel(form,text=lbl).grid(row=r,column=0,sticky="w",padx=4,pady=2)
            wid.grid(row=r,column=1,sticky="w",padx=4,pady=2)

        add_row("元画像", ctk.CTkButton(form,text="選択", command=lambda:self.pick("original_image_path",True)))
        add_row("タイルフォルダ", ctk.CTkButton(form,text="選択", command=lambda:self.pick("tile_images_folder",False)))

        def num_entry(key, is_float=False):
            vcmd = (self.register(lambda s,flt=is_float: re.fullmatch(r'^\d*' + (r'(\.\d*)?' if flt else '') + r'$',s) is not None), "%P")
            e = ctk.CTkEntry(form, width=90, textvariable=self.vars[key], validate="key", validatecommand=vcmd)
            return e
        for k,l,is_f in [
            ("num_title","分割数",False), ("resize_base_px","リサイズ基準",False),
            ("max_tile_usage","同一画像最大使用数",False), ("repeat_limit","連続使用制限",False),
            ("vertical_limit","縦方向制限",False), ("horizontal_limit","横方向制限",False),
            ("brightness_factor","明るさ係数",True), ("color_factor","色合い係数",True),
            ("blur_ksize","Blur ksize(奇数)",False), ("dpi","DPI",False)
            #,("jpeg_quality","JPEG品質",False)
        ]: add_row(l, num_entry(k,is_f))

        for k,l in [("link_vertical_horizontal","縦横連携"),
                    ("brightness_adjust_on","明るさ補正 ON"),
                    ("color_adjust_on","色合い補正 ON"),
                    ("gaussian_blur_on","ガウシアンぼかし ON")]:
            add_row("", ctk.CTkCheckBox(form,text=l,variable=self.bools[k]))

        ctk.CTkButton(form,text="設定を保存",command=self.save_cfg).grid(pady=8,columnspan=2)
        ctk.CTkButton(form,text="RGBリスト生成",command=self.start_rgb_gen).grid(pady=4,columnspan=2)
        ctk.CTkButton(form,text="モザイク生成開始",command=self.start_job).grid(pady=4,columnspan=2)

        self.preview_box=ctk.CTkFrame(self,width=self.PREVIEW_BOX[0],height=self.PREVIEW_BOX[1])
        self.preview_box.grid_propagate(False)
        self.preview_box.grid(row=0,column=1,padx=(5,10),pady=10)
        self.preview_lbl=ctk.CTkLabel(self.preview_box,text=""); self.preview_lbl.pack(expand=True)

        self.zoom=tk.DoubleVar(value=1.0)
        ctk.CTkSlider(self,from_=0.5,to=3,number_of_steps=25,variable=self.zoom,
                      command=lambda *_:self.refresh()).grid(row=1,column=1,sticky="ew",padx=(5,10))
        ctk.CTkLabel(self, text="ズーム").grid(row=1, column=1, sticky="w", padx=(5, 50))

        self.prog=ctk.CTkProgressBar(self); self.prog.grid(row=2,column=0,columnspan=2,sticky="ew",padx=10)
        self.log=tk.StringVar(); ctk.CTkLabel(self,textvariable=self.log)\
            .grid(row=3,column=0,columnspan=2,sticky="w",padx=12,pady=(0,8))

        for k in ["original_image_path","brightness_factor","color_factor","blur_ksize"]:
            self.vars[k].trace_add("write", lambda *_: self.refresh())
        for b in self.bools.values(): b.trace_add("write", lambda *_: self.refresh())

        self.refresh()

    def pick(self,key, file): 
        path = filedialog.askopenfilename() if file else filedialog.askdirectory()
        if path: self.vars[key].set(path)

    def cfg(self):
        cfg={k:(v.get() if k not in self.bools else bool(self.bools[k].get()))
             for k,v in self.vars.items()}
        for k,b in self.bools.items(): cfg[k]=bool(b.get()); return cfg

    def save_cfg(self):
        CONF_FILE.write_text(yaml.safe_dump(self.cfg(),allow_unicode=True),"utf-8")
        messagebox.showinfo("保存","設定を保存しました")

    def refresh(self):
        path = Path(self.vars["original_image_path"].get())
        if not path.is_file():
            self.preview_lbl.configure(text="元画像未選択",image=None); return
        img = Image.open(path).convert("RGB")
        if self.bools["brightness_adjust_on"].get():
            img = ImageEnhance.Brightness(img).enhance(float(self.vars["brightness_factor"].get() or 1))
        if self.bools["color_adjust_on"].get():
            img = ImageEnhance.Color(img).enhance(float(self.vars["color_factor"].get() or 1))
        if self.bools["gaussian_blur_on"].get():
            k=int(self.vars["blur_ksize"].get() or 1)|1
            img=img.filter(ImageFilter.GaussianBlur(radius=k))
        bw,bh=self.PREVIEW_BOX; w,h=img.size
        ratio=min(bw/w,bh/h)*self.zoom.get(); img=img.resize((int(w*ratio),int(h*ratio)))
        cimg=CTkImage(light_image=img,size=img.size)
        self.preview_lbl.configure(image=cimg,text=""); self.preview_lbl.image=cimg

    def start_job(self):
        if not Path(self.vars["original_image_path"].get()).is_file():
            messagebox.showerror("エラー","元画像を選択してください"); return
        if not Path(self.vars["tile_images_folder"].get()).is_dir():
            messagebox.showerror("エラー","タイルフォルダを選択してください"); return
        self.save_cfg(); self.prog.set(0); self.log.set("モザイク生成中…")
        threading.Thread(target=self._run_mosaic, daemon=True).start()

    def start_rgb_gen(self):
        if not Path(self.vars["tile_images_folder"].get()).is_dir():
            messagebox.showerror("エラー","タイルフォルダを選択してください"); return
        self.save_cfg(); self.prog.set(0); self.log.set("RGBリスト生成中…")
        threading.Thread(target=self._run_rgb_gen, daemon=True).start()

    def _run_mosaic(self):
        try:
            generate_mosaic_image()
            self.prog.set(1)
            self.log.set("モザイク生成完了！")
        except Exception as e:
            self.log.set(f"エラー: {e}")

    def _run_rgb_gen(self):
        try:
            save_rgb_values_parallel()
            self.prog.set(1)
            self.log.set("RGBリスト生成完了！")
        except Exception as e:
            self.log.set(f"エラー: {e}")

# ── 起動 ───────────────────────────────────────
if __name__=="__main__":
    ctk.set_appearance_mode("System"); ctk.set_default_color_theme("blue")
    MosaicApp().mainloop()

'''''
