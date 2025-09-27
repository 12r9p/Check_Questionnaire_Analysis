import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, scrolledtext
import json

try:
    from PIL import Image, ImageTk
except ImportError:
    messagebox.showerror(
        "ライブラリ不足", 
        "このアプリケーションにはPillowライブラリが必要です。\n"
        "ターミナルで `pip install Pillow` を実行してインストールしてください。"
    )
    exit()

class CoordinateSetterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("座標設定ツール (ズーム: スクロール, 移動: Shift+ドラッグ)")
        self.geometry("1200x800")

        # --- データ ---
        self.image_path = None
        self.original_image = None
        self.photo_image = None
        self.regions = [] # {name, bbox, type} のリスト
        
        # --- レイアウト ---
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Button(top_frame, text="画像ファイルを開く", command=self.load_image).pack(side=tk.LEFT)
        
        self.mode = tk.StringVar(value="checkbox")
        tk.Radiobutton(top_frame, text="チェックボックス", variable=self.mode, value="checkbox").pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(top_frame, text="自由記述欄", variable=self.mode, value="free_text").pack(side=tk.LEFT)

        main_frame = tk.Frame(self)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main_frame, bg="#2d2d2d")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_panel = tk.Frame(main_frame, width=250)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        tk.Label(right_panel, text="設定済み領域リスト").pack()
        self.region_listbox = tk.Listbox(right_panel)
        self.region_listbox.pack(fill=tk.BOTH, expand=True)
        
        tk.Button(right_panel, text="選択した領域を削除", command=self.delete_selected_region).pack(fill=tk.X, pady=5)

        bottom_frame = tk.Frame(self, height=200)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        tk.Button(bottom_frame, text="設定コードを生成", command=self.generate_config_code).pack()
        self.code_text = scrolledtext.ScrolledText(bottom_frame, height=10, wrap=tk.WORD)
        self.code_text.pack(fill=tk.BOTH, expand=True)

        # --- イベントバインド ---
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        # パン（移動）
        self.canvas.bind("<Shift-ButtonPress-1>", self.pan_start)
        self.canvas.bind("<Shift-B1-Motion>", self.pan_move)
        # ズーム
        self.canvas.bind("<MouseWheel>", self.zoom) # for Windows, macOS
        self.canvas.bind("<Button-4>", self.zoom) # for Linux
        self.canvas.bind("<Button-5>", self.zoom) # for Linux
        
        self.start_x = self.start_y = 0
        self.current_rect = None
        self.scale = 1.0

    def load_image(self):
        path = filedialog.askopenfilename(
            title="画像ファイルを選択",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if not path:
            return
            
        self.image_path = path
        self.original_image = Image.open(self.image_path)
        self.scale = 1.0
        self.regions = []
        self.region_listbox.delete(0, tk.END)
        self.code_text.delete(1.0, tk.END)
        self.show_image()

    def show_image(self):
        if not self.original_image: return

        w, h = self.original_image.size
        disp_w, disp_h = int(w * self.scale), int(h * self.scale)
        
        disp_image = self.original_image.resize((disp_w, disp_h), Image.Resampling.NEAREST)
        self.photo_image = ImageTk.PhotoImage(disp_image)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image, tags="image")
        self.draw_regions()

    def on_press(self, event):
        if not self.original_image: return
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, 
            outline="red", dash=(2, 2), tags="temp_rect"
        )

    def on_drag(self, event):
        if not self.current_rect: return
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.current_rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_release(self, event):
        if not self.current_rect: return
        
        self.canvas.delete("temp_rect")
        self.current_rect = None
        
        # スケールを考慮して元の画像座標に変換
        x1 = min(self.start_x, self.canvas.canvasx(event.x)) / self.scale
        y1 = min(self.start_y, self.canvas.canvasy(event.y)) / self.scale
        x2 = max(self.start_x, self.canvas.canvasx(event.x)) / self.scale
        y2 = max(self.start_y, self.canvas.canvasy(event.y)) / self.scale
        
        if abs(x1 - x2) < 3 or abs(y1 - y2) < 3: return

        name = simpledialog.askstring("ラベル名", "この領域のラベル名を入力してください:", parent=self)
        if name:
            bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            self.regions.append({"name": name, "bbox": bbox, "type": self.mode.get()})
            self.region_listbox.insert(tk.END, f"{name} ({self.mode.get()})")
            self.draw_regions()

    def draw_regions(self):
        self.canvas.delete("region")
        for region in self.regions:
            x, y, w, h = region["bbox"]
            # 元の画像座標を現在のスケールに合わせて描画
            disp_x1 = x * self.scale
            disp_y1 = y * self.scale
            disp_x2 = (x + w) * self.scale
            disp_y2 = (y + h) * self.scale
            
            color = "cyan" if region["type"] == "checkbox" else "magenta"
            self.canvas.create_rectangle(disp_x1, disp_y1, disp_x2, disp_y2, outline=color, width=2, tags="region")
            self.canvas.create_text(disp_x1, disp_y1 - 5, text=region["name"], anchor=tk.SW, fill=color, tags="region")

    def delete_selected_region(self):
        selected_indices = self.region_listbox.curselection()
        if not selected_indices: return
        for index in reversed(selected_indices):
            self.region_listbox.delete(index)
            del self.regions[index]
        self.draw_regions()

    def generate_config_code(self):
        # ... (この関数は変更なし)
        checkboxes = []
        free_texts = []
        for region in self.regions:
            item = {"name": region["name"], "bbox": region["bbox"]}
            if region["type"] == "checkbox":
                checkboxes.append(item)
            else:
                item["threshold"] = 0.01 
                free_texts.append(item)
        config = {"checkboxes": checkboxes, "free_texts": free_texts, "checkbox_threshold": 0.1}
        config_str = "CONFIG = " + json.dumps(config, indent=4, ensure_ascii=False)
        self.code_text.delete(1.0, tk.END)
        self.code_text.insert(tk.END, config_str)
        messagebox.showinfo("成功", "設定コードが生成されました。下のボックスからコピーしてください。")

    def pan_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def pan_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def zoom(self, event):
        if event.delta > 0 or event.num == 4: # Zoom in
            factor = 1.1
        elif event.delta < 0 or event.num == 5: # Zoom out
            factor = 0.9
        else:
            return
            
        self.scale *= factor
        # ズームの中心をカーソル位置に
        self.canvas.scale("all", event.x, event.y, factor, factor)
        self.draw_regions() # スケール変更後に再描画

if __name__ == "__main__":
    app = CoordinateSetterApp()
    app.mainloop()