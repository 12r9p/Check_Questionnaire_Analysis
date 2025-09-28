# --- 実行に必要なライブラリ ---
# このスクリプトを実行するには、以下のライブラリが必要です。
# pipやcondaでインストールしてください。
# pip install opencv-python pandas matplotlib japanize-matplotlib tqdm

import os
import shutil
import glob
import cv2
import pandas as pd
import numpy as np
import json
from tqdm import tqdm # tqdmをインポート

# --- 初期設定 ---
IMAGE_DIR = "./images"
OUTPUT_DIR = "./free_texts"
DEBUG_DIR = "./debug_output" # デバッグ画像用のフォルダ
OUTPUT_CSV = "results.csv"
CONFIG_PATH = "config.json"

# --- 設定ファイルの読み込み ---
CONFIG = {}
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
    print(f"✅ {CONFIG_PATH} を読み込みました。")
except FileNotFoundError:
    print(f"エラー: {CONFIG_PATH} が見つかりません。coordinate_setter.pyで作成してください。")
    exit()
except json.JSONDecodeError:
    print(f"エラー: {CONFIG_PATH} のフォーマットが正しくありません。")
    exit()

# --- ディレクトリ作成 ---
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEBUG_MODE = CONFIG.get("debug_mode", False)
if DEBUG_MODE:
    os.makedirs(DEBUG_DIR, exist_ok=True)
    print("✅ デバッグモードが有効です。処理画像が 'debug_output' に保存されます。")

# --- 解析用コア関数 ---
def get_black_pixel_ratio(binary_roi):
    """事前に二値化されたROI画像内の黒ピクセル（値が0でないピクセル）の割合を計算する。"""
    if binary_roi.size == 0: return 0.0
    return np.count_nonzero(binary_roi) / binary_roi.size

def is_checked(binary_roi, threshold):
    """二値化ROIに基づいてチェックされているかを判定する。"""
    ratio = get_black_pixel_ratio(binary_roi)
    return ratio >= threshold, ratio

def has_text(binary_roi, threshold):
    """二値化ROIに基づいてテキストが存在するかを判定する。"""
    return get_black_pixel_ratio(binary_roi) >= threshold

# --- 自動座標補正（アフィン変換）関数 ---
def get_affine_transform(image, anchors_config):
    if not anchors_config or len(anchors_config) != 2:
        return None, "アンカーが2つ定義されていません。"

    src_points, dst_points = [], []
    img_h, img_w, _ = image.shape
    # 探索範囲のマージン。スキャン画像の位置ズレが少ないことを想定した最適化。
    # この値を大きくすると探索範囲は広がるが、処理時間は長くなる。
    SEARCH_MARGIN = 100 

    for anchor in anchors_config:
        try:
            template_path = anchor["template_path"]
            if not os.path.exists(template_path):
                return None, f"アンカー画像が見つかりません: {template_path}"
            template = cv2.imread(template_path)
            if template is None: return None, f"アンカー画像の読み込みに失敗: {template_path}"
            
            h, w, _ = template.shape
            ex, ey, ew, eh = anchor["expected_bbox"]

            # 探索範囲(ROI)を定義
            roi_x = max(0, ex - SEARCH_MARGIN)
            roi_y = max(0, ey - SEARCH_MARGIN)
            roi_w = min(img_w - roi_x, ew + 2 * SEARCH_MARGIN)
            roi_h = min(img_h - roi_y, eh + 2 * SEARCH_MARGIN)
            
            # 画像が探索範囲より小さい場合のフォールバック
            if roi_w < w or roi_h < h:
                search_image = image
                roi_x, roi_y = 0, 0
            else:
                search_image = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

            res = cv2.matchTemplate(search_image, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc_relative = cv2.minMaxLoc(res)

            if max_val < 0.7: # マッチングの信頼度閾値
                return None, f'{anchor["name"]} の信頼性が低すぎます ({max_val:.2f})。探索範囲を広げる必要があるかもしれません。'

            # ROIの左上座標を足して、画像全体の座標に変換
            max_loc = (max_loc_relative[0] + roi_x, max_loc_relative[1] + roi_y)

            src_points.append([max_loc[0] + w / 2, max_loc[1] + h / 2])
            dst_points.append([ex + ew / 2, ey + eh / 2])
        except Exception as e:
            return None, f"アンカー処理中にエラー: {e}"

    p1_src, p2_src = src_points
    p1_dst, p2_dst = dst_points
    dx_src, dy_src = p2_src[0] - p1_src[0], p2_src[1] - p1_src[1]
    dx_dst, dy_dst = p2_dst[0] - p1_dst[0], p2_dst[1] - p1_dst[1]
    p3_src = [p1_src[0] - dy_src, p1_src[1] + dx_src]
    p3_dst = [p1_dst[0] - dy_dst, p1_dst[1] + dx_dst]
    
    src_pts_np = np.float32([p1_src, p2_src, p3_src])
    dst_pts_np = np.float32([p1_dst, p2_dst, p3_dst])

    return cv2.getAffineTransform(src_pts_np, dst_pts_np), "成功"

# --- 解析実行 --- 
def run_analysis(config, image_dir, output_dir, output_csv, debug_dir):
    all_results = []
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))

    if not image_paths: return print(f"⚠️ 解析対象の画像が {image_dir} に見つかりません。")
    print(f"解析対象: {len(image_paths)}枚の画像")

    for path in tqdm(image_paths, desc="解析処理中", unit="枚"):
        img = cv2.imread(path)
        if img is None: continue

        corrected_img = img
        matrix, status = get_affine_transform(img, config.get("anchors"))
        if matrix is not None:
            rows, cols, _ = img.shape
            corrected_img = cv2.warpAffine(img, matrix, (cols, rows))
        # 補正に失敗した場合はコンソールに警告を出す
        elif status != "アンカーが2つ定義されていません。":
             tqdm.write(f"  -> 警告: {os.path.basename(path)} の座標補正をスキップ。理由: {status}")

        debug_image = corrected_img.copy() if DEBUG_MODE else None
        result = {"file_path": os.path.basename(path)}

        # --- 高速化: 画像全体を事前にグレースケール化・二値化 ---
        gray_img = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # --- デバッグ: マーカーの期待位置を描画 ---
        if DEBUG_MODE and config.get("anchors"):
            for anchor in config.get("anchors"):
                ex, ey, ew, eh = anchor["expected_bbox"]
                # 青色でマーカーの期待位置を囲む
                cv2.rectangle(debug_image, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                cv2.putText(debug_image, anchor["name"], (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        for cb in config.get("checkboxes", []):
            x, y, w, h = cb["bbox"]
            # 事前に二値化した画像からROIを切り出す
            binary_roi = binary_img[y:y+h, x:x+w]
            checked, ratio = is_checked(binary_roi, cb["threshold"])
            result[cb["name"]] = int(checked)
            if DEBUG_MODE:
                color = (0, 255, 0) if checked else (0, 0, 255)
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), color, 2)
                cv2.putText(debug_image, f'{ratio:.5f}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        any_text_found = False
        for ft in config.get("free_texts", []):
            fx, fy, fw, fh = ft["bbox"]
            # 事前に二値化した画像からROIを切り出す
            binary_roi = binary_img[fy:fy+fh, fx:fx+fw]
            text_found = has_text(binary_roi, ft["threshold"])
            result[ft["name"]] = int(text_found)
            if text_found: any_text_found = True
            if DEBUG_MODE:
                color = (0, 255, 0) if text_found else (0, 0, 255)
                cv2.rectangle(debug_image, (fx, fy), (fx+fw, fy+fh), color, 2)

        if any_text_found:
            basename = os.path.basename(path)
            copy_path = os.path.join(output_dir, basename)
            shutil.copy(path, copy_path)
            result["free_text_image_copied"] = os.path.basename(copy_path)
        else:
            result["free_text_image_copied"] = ""
        
        all_results.append(result)

        if DEBUG_MODE:
            debug_path = os.path.join(debug_dir, os.path.basename(path))
            cv2.imwrite(debug_path, debug_image)

    if not all_results: return print("解析結果がありませんでした。")

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ 全画像解析完了。結果を {output_csv} に出力しました。")
    if DEBUG_MODE: print(f"✅ デバッグ画像が {debug_dir} に保存されました。")

# --- メイン処理フロー ---
if __name__ == "__main__":
    print("--- アンケート解析スクリプト ---")
    run_analysis(CONFIG, IMAGE_DIR, OUTPUT_DIR, OUTPUT_CSV, DEBUG_DIR)
