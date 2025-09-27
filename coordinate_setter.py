
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, scrolledtext
import json
import copy

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
        self.title("座標設定ツール (ズーム: スクロール, 移動: Shift+ドラッグ, 領域移動: ドラッグ)")
        self.geometry("1200x800")

        # --- データ ---
        self.original_image = None
        self.photo_image = None
        self.regions = []
        self.scale = 1.0
        
        # --- 状態管理 ---
        self.drag_data = {"x": 0, "y": 0, "item": None, "mode": None} # mode: "draw" or "move"

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
        
        btn_frame = tk.Frame(right_panel)
        btn_frame.pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="複製", command=self.duplicate_selected_region).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(btn_frame, text="削除", command=self.delete_selected_region).pack(side=tk.LEFT, expand=True, fill=tk.X)

        bottom_frame = tk.Frame(self, height=200)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        tk.Button(bottom_frame, text="設定コードを生成", command=self.generate_config_code).pack()
        self.code_text = scrolledtext.ScrolledText(bottom_frame, height=10, wrap=tk.WORD)
        self.code_text.pack(fill=tk.BOTH, expand=True)

        # --- イベントバインド ---
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Shift-ButtonPress-1>", self.pan_start)
        self.canvas.bind("<Shift-B1-Motion>", self.pan_move)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-4>", self.zoom)
        self.canvas.bind("<Button-5>", self.zoom)
        self.region_listbox.bind("<Double-Button-1>", self.edit_region_name)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if not path: return
        self.original_image = Image.open(path)
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
        canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        
        # Find if a region is clicked
        clicked_region_index = self.get_region_at(canvas_x, canvas_y)
        if clicked_region_index is not None:
            self.drag_data["mode"] = "move"
            self.drag_data["index"] = clicked_region_index
            self.drag_data["x"] = canvas_x
            self.drag_data["y"] = canvas_y
            self.region_listbox.selection_clear(0, tk.END)
            self.region_listbox.selection_set(clicked_region_index)
        else:
            self.drag_data["mode"] = "draw"
            self.drag_data["x"] = canvas_x
            self.drag_data["y"] = canvas_y
            self.drag_data["item"] = self.canvas.create_rectangle(canvas_x, canvas_y, canvas_x, canvas_y, outline="red", dash=(2, 2), tags="temp_rect")

    def on_drag(self, event):
        if not self.original_image: return
        canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        if self.drag_data["mode"] == "draw" and self.drag_data["item"]:
            self.canvas.coords(self.drag_data["item"], self.drag_data["x"], self.drag_data["y"], canvas_x, canvas_y)
        elif self.drag_data["mode"] == "move":
            dx = canvas_x - self.drag_data["x"]
            dy = canvas_y - self.drag_data["y"]
            region_index = self.drag_data["index"]
            region = self.regions[region_index]
            region["bbox"][0] += dx / self.scale
            region["bbox"][1] += dy / self.scale
            self.drag_data["x"], self.drag_data["y"] = canvas_x, canvas_y
            self.draw_regions()

    def on_release(self, event):
        if self.drag_data["mode"] == "draw" and self.drag_data["item"]:
            self.canvas.delete("temp_rect")
            x1 = min(self.drag_data["x"], self.canvas.canvasx(event.x)) / self.scale
            y1 = min(self.drag_data["y"], self.canvas.canvasy(event.y)) / self.scale
            x2 = max(self.drag_data["x"], self.canvas.canvasx(event.x)) / self.scale
            y2 = max(self.drag_data["y"], self.canvas.canvasy(event.y)) / self.scale
            if abs(x1 - x2) < 3 or abs(y1 - y2) < 3: return
            name = simpledialog.askstring("ラベル名", "この領域のラベル名を入力してください:", parent=self)
            if name:
                bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                self.regions.append({"name": name, "bbox": bbox, "type": self.mode.get()})
                self.update_listbox()
                self.draw_regions()
        elif self.drag_data["mode"] == "move":
            # 移動完了時に、座標を整数に確定させる
            region_index = self.drag_data["index"]
            region = self.regions[region_index]
            region["bbox"][0] = int(region["bbox"][0])
            region["bbox"][1] = int(region["bbox"][1])
            # 念のため、幅と高さも整数に
            region["bbox"][2] = int(region["bbox"][2])
            region["bbox"][3] = int(region["bbox"][3])
            self.draw_regions() # 最終位置を再描画

        self.drag_data["mode"] = None
        self.drag_data["item"] = None

    def get_region_at(self, x, y):
        for i, region in reversed(list(enumerate(self.regions))):
            r_x, r_y, r_w, r_h = region["bbox"]
            disp_x1, disp_y1 = r_x * self.scale, r_y * self.scale
            disp_x2, disp_y2 = (r_x + r_w) * self.scale, (r_y + r_h) * self.scale
            if disp_x1 <= x <= disp_x2 and disp_y1 <= y <= disp_y2:
                return i
        return None

    def draw_regions(self):
        self.canvas.delete("region")
        for region in self.regions:
            x, y, w, h = region["bbox"]
            disp_x1, disp_y1 = x * self.scale, y * self.scale
            disp_x2, disp_y2 = (x + w) * self.scale, (y + h) * self.scale
            color = "cyan" if region["type"] == "checkbox" else "magenta"
            self.canvas.create_rectangle(disp_x1, disp_y1, disp_x2, disp_y2, outline=color, width=2, tags="region")
            self.canvas.create_text(disp_x1, disp_y1 - 5, text=region["name"], anchor=tk.SW, fill=color, tags="region")

    def update_listbox(self):
        self.region_listbox.delete(0, tk.END)
        for region in self.regions:
            self.region_listbox.insert(tk.END, f'{region["name"]} ({region["type"]})')

    def delete_selected_region(self):
        selected_indices = self.region_listbox.curselection()
        if not selected_indices: return
        for index in reversed(selected_indices):
            del self.regions[index]
        self.update_listbox()
        self.draw_regions()

    def duplicate_selected_region(self):
        selected_indices = self.region_listbox.curselection()
        if not selected_indices: return
        for index in selected_indices:
            original_region = self.regions[index]
            new_region = copy.deepcopy(original_region)
            new_region["name"] = f'{original_region["name"]}_copy'
            new_region["bbox"][0] += 10 # 少しずらす
            new_region["bbox"][1] += 10
            self.regions.append(new_region)
        self.update_listbox()
        self.draw_regions()

    def edit_region_name(self, event):
        selected_indices = self.region_listbox.curselection()
        if not selected_indices: return
        index = selected_indices[0]
        region = self.regions[index]
        old_name = region["name"]
        new_name = simpledialog.askstring("ラベル名変更", "新しいラベル名を入力してください:", initialvalue=old_name, parent=self)
        if new_name and new_name != old_name:
            region["name"] = new_name
            self.update_listbox()
            self.draw_regions()

    def generate_config_code(self):
        checkboxes, free_texts = [], []
        for region in self.regions:
            name_str, bbox_str = repr(region["name"]), str(region["bbox"])
            if region["type"] == "checkbox":
                threshold = 0.1
                checkboxes.append(f'        {{"name": {name_str}, "bbox": {bbox_str}, "threshold": {threshold}}}')
            else:
                threshold = 0.01
                free_texts.append(f'        {{"name": {name_str}, "bbox": {bbox_str}, "threshold": {threshold}}}')
        config_parts = ["CONFIG = {"]
        config_parts.append('    "checkboxes": [')
        config_parts.append(",\n".join(checkboxes))
        config_parts.append('    ],')
        config_parts.append('    "free_texts": [')
        config_parts.append(",\n".join(free_texts))
        config_parts.append('    ]')
        config_parts.append("}")
        config_str = "\n".join(config_parts)
        self.code_text.delete(1.0, tk.END)
        self.code_text.insert(tk.END, config_str)
        messagebox.showinfo("成功", "設定コードが生成されました。下のボックスからコピーしてください。")

    def pan_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def pan_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def zoom(self, event):
        if event.delta > 0 or event.num == 4:
            factor = 1.1
        elif event.delta < 0 or event.num == 5:
            factor = 0.9
        else:
            return
        self.scale *= factor
        self.show_image()

if __name__ == "__main__":
    app = CoordinateSetterApp()
    app.mainloop()
