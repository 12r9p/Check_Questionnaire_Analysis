

import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import json
import copy
import os

try:
    from PIL import Image, ImageTk
except ImportError:
    messagebox.showerror("ライブラリ不足", "Pillowライブラリが必要です。`pip install Pillow`でインストールしてください。")
    exit()

class CoordinateSetterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("座標設定ツール")
        self.geometry("1200x800")

        self.original_image = None
        self.photo_image = None
        self.regions = []
        self.scale = 1.0
        self.image_dir = "."
        self.drag_data = {"x": 0, "y": 0, "item": None, "mode": None}

        # --- Top Control Frames ---
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        tk.Button(top_frame, text="画像を開く", command=self.load_image).pack(side=tk.LEFT)
        tk.Button(top_frame, text="設定をJSONから読み込む", command=self.load_config_from_json).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="設定をJSONに保存", command=self.save_config_to_json).pack(side=tk.LEFT)

        mode_frame = tk.Frame(self)
        mode_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.mode = tk.StringVar(value="checkbox")
        tk.Radiobutton(mode_frame, text="チェックボックス", variable=self.mode, value="checkbox").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(mode_frame, text="自由記述欄", variable=self.mode, value="free_text").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(mode_frame, text="アンカー1", variable=self.mode, value="anchor1").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(mode_frame, text="アンカー2", variable=self.mode, value="anchor2").pack(side=tk.LEFT, padx=5)
        
        self.debug_mode_var = tk.BooleanVar(value=True)
        tk.Checkbutton(mode_frame, text="デバッグモード有効", variable=self.debug_mode_var).pack(side=tk.RIGHT, padx=10)

        bulk_edit_frame = tk.Frame(self)
        bulk_edit_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        tk.Label(bulk_edit_frame, text="チェックボックス閾値の一括変更:").pack(side=tk.LEFT)
        self.bulk_threshold_var = tk.StringVar(value="0.1")
        tk.Entry(bulk_edit_frame, textvariable=self.bulk_threshold_var, width=5).pack(side=tk.LEFT, padx=5)
        tk.Button(bulk_edit_frame, text="一括適用", command=self.apply_bulk_threshold).pack(side=tk.LEFT)

        # --- Main Content Frame ---
        main_frame = tk.Frame(self)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main_frame, bg="#2d2d2d")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Right Panel ---
        right_panel = tk.Frame(main_frame, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        tk.Label(right_panel, text="設定済み領域リスト").pack()
        self.region_listbox = tk.Listbox(right_panel)
        self.region_listbox.pack(fill=tk.BOTH, expand=True)
        self.region_listbox.bind("<<ListboxSelect>>", self.on_region_select)
        
        btn_frame = tk.Frame(right_panel)
        btn_frame.pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="複製", command=self.duplicate_selected_region).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(btn_frame, text="削除", command=self.delete_selected_region).pack(side=tk.LEFT, expand=True, fill=tk.X)

        # --- Edit Panel ---
        self.edit_panel = tk.Frame(right_panel, borderwidth=1, relief=tk.GROOVE)
        self.edit_panel.pack(fill=tk.X, pady=10)
        tk.Label(self.edit_panel, text="選択項目の編集").pack()
        
        self.edit_fields = {}
        for field in ["name", "threshold"]:
            row = tk.Frame(self)
            row.pack(fill=tk.X, padx=5, pady=2)
            tk.Label(row, text=field, width=10, anchor=tk.W).pack(side=tk.LEFT)
            self.edit_fields[field] = tk.StringVar()
            tk.Entry(row, textvariable=self.edit_fields[field]).pack(side=tk.RIGHT, expand=True, fill=tk.X)
        
        tk.Button(self.edit_panel, text="更新", command=self.update_selected_region).pack(pady=5)

        # --- Event Bindings ---
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Shift-ButtonPress-1>", self.pan_start)
        self.canvas.bind("<Shift-B1-Motion>", self.pan_move)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-4>", self.zoom)
        self.canvas.bind("<Button-5>", self.zoom)

    # ... (rest of the methods will be updated or added) ...

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if not path: return
        self.image_dir = os.path.dirname(path)
        self.original_image = Image.open(path)
        self.scale = 1.0
        self.regions = []
        self.update_listbox()
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
        clicked_region_index = self.get_region_at(canvas_x, canvas_y)
        if clicked_region_index is not None:
            self.drag_data["mode"] = "move"
            self.drag_data["index"] = clicked_region_index
            self.drag_data["x"] = canvas_x
            self.drag_data["y"] = canvas_y
            self.region_listbox.selection_clear(0, tk.END)
            self.region_listbox.selection_set(clicked_region_index)
            self.on_region_select(None) # Update edit panel
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
            region = self.regions[self.drag_data["index"]]
            region["bbox"][0] += dx / self.scale
            region["bbox"][1] += dy / self.scale
            self.drag_data["x"], self.drag_data["y"] = canvas_x, canvas_y
            self.draw_regions()

    def on_release(self, event):
        mode = self.drag_data["mode"]
        if mode == "draw" and self.drag_data["item"]:
            self.canvas.delete("temp_rect")
            x1, y1 = min(self.drag_data["x"], self.canvas.canvasx(event.x)), min(self.drag_data["y"], self.canvas.canvasy(event.y))
            x2, y2 = max(self.drag_data["x"], self.canvas.canvasx(event.x)), max(self.drag_data["y"], self.canvas.canvasy(event.y))
            if abs(x1 - x2) < 5 or abs(y1 - y2) < 5: return
            self.add_region(x1 / self.scale, y1 / self.scale, x2 / self.scale, y2 / self.scale)
        elif mode == "move":
            region = self.regions[self.drag_data["index"]]
            for i in range(4): region["bbox"][i] = int(region["bbox"][i])
            self.draw_regions()
        self.drag_data["mode"] = None
        self.drag_data["item"] = None

    def add_region(self, x1, y1, x2, y2):
        region_type = self.mode.get()
        name = region_type
        if region_type not in ["anchor1", "anchor2"]:
            name = simpledialog.askstring("ラベル名", "この領域のラベル名を入力してください:", parent=self)
            if not name: return
        else: 
            self.regions = [r for r in self.regions if r["type"] != region_type]

        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        new_region = {"name": name, "bbox": bbox, "type": region_type}
        if region_type in ["checkbox", "free_text"]:
            new_region["threshold"] = 0.1 if region_type == "checkbox" else 0.01
        self.regions.append(new_region)
        self.update_listbox()
        self.draw_regions()

        if region_type in ["anchor1", "anchor2"]:
            try:
                anchor_img = self.original_image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
                save_path = os.path.join(self.image_dir, f"{region_type}.png")
                anchor_img.save(save_path)
                messagebox.showinfo("アンカー保存", f"{region_type}を\n{save_path}\nに保存しました。")
            except Exception as e:
                messagebox.showerror("アンカー保存失敗", f"エラー: {e}")

    def get_region_at(self, x, y):
        for i, region in reversed(list(enumerate(self.regions))):
            r_x, r_y, r_w, r_h = region["bbox"]
            disp_x1, disp_y1, disp_x2, disp_y2 = r_x * self.scale, r_y * self.scale, (r_x + r_w) * self.scale, (r_y + r_h) * self.scale
            if disp_x1 <= x <= disp_x2 and disp_y1 <= y <= disp_y2: return i
        return None

    def draw_regions(self):
        self.canvas.delete("region")
        for region in self.regions:
            x, y, w, h = region["bbox"]
            disp_x1, disp_y1, disp_x2, disp_y2 = x * self.scale, y * self.scale, (x + w) * self.scale, (y + h) * self.scale
            color = {"checkbox": "blue", "free_text": "green", "anchor1": "red", "anchor2": "yellow"}.get(region["type"], "white")
            self.canvas.create_rectangle(disp_x1, disp_y1, disp_x2, disp_y2, outline=color, width=2, tags="region")
            self.canvas.create_text(disp_x1, disp_y1 - 5, text=region["name"], anchor=tk.SW, fill=color, tags="region")

    def update_listbox(self):
        self.region_listbox.delete(0, tk.END)
        for region in sorted(self.regions, key=lambda r: r["type"]):
            text = f'{region["name"]} ({region["type"]})'
            if "threshold" in region:
                text += f', t={region["threshold"]}'
            self.region_listbox.insert(tk.END, text)

    def delete_selected_region(self):
        selected_indices = self.region_listbox.curselection()
        if not selected_indices: return
        # Get full text from listbox to find the exact region to delete
        selected_texts = [self.region_listbox.get(i) for i in selected_indices]
        self.regions = [r for r in self.regions if f'{r["name"]} ({r["type"]})' + (f', t={r["threshold"]}' if "threshold" in r else '') not in selected_texts]
        self.update_listbox()
        self.draw_regions()

    def duplicate_selected_region(self):
        selected_indices = self.region_listbox.curselection()
        if not selected_indices: return
        selected_text = self.region_listbox.get(selected_indices[0])
        original_region = next((r for r in self.regions if f'{r["name"]} ({r["type"]})' + (f', t={r["threshold"]}' if "threshold" in r else '') == selected_text), None)
        if original_region and original_region["type"] not in ["anchor1", "anchor2"]:
            new_region = copy.deepcopy(original_region)
            new_region["name"] = f'{original_region["name"]}_copy'
            new_region["bbox"][0] += 10
            new_region["bbox"][1] += 10
            self.regions.append(new_region)
            self.update_listbox()
            self.draw_regions()

    def on_region_select(self, event):
        selected_indices = self.region_listbox.curselection()
        if not selected_indices: return
        selected_text = self.region_listbox.get(selected_indices[0])
        region = next((r for r in self.regions if f'{r["name"]} ({r["type"]})' + (f', t={r["threshold"]}' if "threshold" in r else '') == selected_text), None)
        if region:
            self.edit_fields["name"].set(region.get("name", ""))
            self.edit_fields["threshold"].set(region.get("threshold", ""))
        else:
            self.edit_fields["name"].set("")
            self.edit_fields["threshold"].set("")

    def update_selected_region(self):
        selected_indices = self.region_listbox.curselection()
        if not selected_indices: return
        selected_text = self.region_listbox.get(selected_indices[0])
        region = next((r for r in self.regions if f'{r["name"]} ({r["type"]})' + (f', t={r["threshold"]}' if "threshold" in r else '') == selected_text), None)
        if region:
            try:
                new_name = self.edit_fields["name"].get()
                new_threshold = self.edit_fields["threshold"].get()
                if new_name: region["name"] = new_name
                if new_threshold and region["type"] in ["checkbox", "free_text"]:
                    region["threshold"] = float(new_threshold)
                self.update_listbox()
                self.draw_regions()
            except ValueError:
                messagebox.showerror("入力エラー", "閾値は数値で入力してください。")

    def apply_bulk_threshold(self):
        try:
            new_threshold = float(self.bulk_threshold_var.get())
            for region in self.regions:
                if region["type"] == "checkbox":
                    region["threshold"] = new_threshold
            self.update_listbox()
            self.draw_regions()
            messagebox.showinfo("成功", f"すべてのチェックボックスの閾値が {new_threshold} に更新されました。")
        except ValueError:
            messagebox.showerror("入力エラー", "閾値は数値で入力してください。")

    def save_config_to_json(self):
        if not self.regions: return messagebox.showwarning("警告", "設定が空です。")
        save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], initialdir=self.image_dir, initialfile="config.json", title="設定をJSONファイルに保存")
        if not save_path: return
        
        config_dict = self.build_config_dict()
        
        # --- コンパクトなJSON文字列を手動で構築 ---
        main_parts = []
        main_parts.append(f'    "debug_mode": {str(config_dict["debug_mode"]).lower()}')

        for key in ["anchors", "checkboxes", "free_texts"]:
            block_lines = []
            block_lines.append(f'    "{key}": [')
            items_str = []
            for item in config_dict.get(key, []):
                items_str.append("        " + json.dumps(item, ensure_ascii=False))
            block_lines.append(",\n".join(items_str))
            block_lines.append('    ]')
            main_parts.append("\n".join(block_lines))

        config_str = "{\n" + ",\n".join(main_parts) + "\n}"
        # --- ここまで --- 

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(config_str)
            messagebox.showinfo("成功", f"設定が {os.path.basename(save_path)} に保存されました。")
        except Exception as e:
            messagebox.showerror("保存失敗", f"エラー: {e}")

    def build_config_dict(self):
        anchors, checkboxes, free_texts = [], [], []
        for r in self.regions:
            region_type = r["type"]
            if region_type in ["anchor1", "anchor2"]:
                # 'images' ディレクトリからの相対パスを記録
                path_in_json = os.path.join("images", f'{r["name"]}.png').replace("\\", "/")
                anchors.append({"name": r["name"], "template_path": path_in_json, "expected_bbox": r["bbox"]})
            elif region_type == "checkbox":
                checkboxes.append({"name": r["name"], "bbox": r["bbox"], "threshold": r.get("threshold", 0.1)})
            elif region_type == "free_text":
                free_texts.append({"name": r["name"], "bbox": r["bbox"], "threshold": r.get("threshold", 0.01)})
        return {"debug_mode": self.debug_mode_var.get(), "anchors": sorted(anchors, key=lambda a: a["name"]), "checkboxes": checkboxes, "free_texts": free_texts}

    def load_config_from_json(self):
        if not self.original_image: return messagebox.showwarning("警告", "先に画像を開いてください。")
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")], initialdir=self.image_dir, title="設定ファイルを読み込む")
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.regions = []
            self.debug_mode_var.set(config.get("debug_mode", False))
            for item in config.get("checkboxes", []): self.regions.append({"name": item["name"], "bbox": item["bbox"], "type": "checkbox", "threshold": item.get("threshold", 0.1)})
            for item in config.get("free_texts", []): self.regions.append({"name": item["name"], "bbox": item["bbox"], "type": "free_text", "threshold": item.get("threshold", 0.01)})
            for item in config.get("anchors", []): self.regions.append({"name": item["name"], "bbox": item["expected_bbox"], "type": item["name"]})
            self.update_listbox()
            self.draw_regions()
            messagebox.showinfo("成功", "設定をファイルから読み込みました。")
        except Exception as e:
            messagebox.showerror("読み込み失敗", f"エラー: {e}")

    def pan_start(self, event): self.canvas.scan_mark(event.x, event.y)
    def pan_move(self, event): self.canvas.scan_dragto(event.x, event.y, gain=1)
    def zoom(self, event):
        factor = 1.1 if (event.delta > 0 or event.num == 4) else 0.9
        self.scale *= factor
        self.show_image()

if __name__ == "__main__":
    app = CoordinateSetterApp()
    app.mainloop()
