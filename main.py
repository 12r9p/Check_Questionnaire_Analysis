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
    # 戻り値に src_pts_np と dst_pts_np を追加
    if not anchors_config or len(anchors_config) != 2:
        return None, "アンカーが2つ定義されていません。", None, None

    src_points, dst_points = [], []
    img_h, img_w, _ = image.shape
    SEARCH_MARGIN = 100 

    for anchor in anchors_config:
        try:
            template_path = anchor["template_path"]
            if not os.path.exists(template_path):
                return None, f"アンカー画像が見つかりません: {template_path}", None, None
            template = cv2.imread(template_path)
            if template is None: return None, f"アンカー画像の読み込みに失敗: {template_path}", None, None
            
            h, w, _ = template.shape
            ex, ey, ew, eh = anchor["expected_bbox"]

            roi_x = max(0, ex - SEARCH_MARGIN)
            roi_y = max(0, ey - SEARCH_MARGIN)
            roi_w = min(img_w - roi_x, ew + 2 * SEARCH_MARGIN)
            roi_h = min(img_h - roi_y, eh + 2 * SEARCH_MARGIN)
            
            if roi_w < w or roi_h < h:
                search_image = image
                roi_x, roi_y = 0, 0
            else:
                search_image = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

            res = cv2.matchTemplate(search_image, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc_relative = cv2.minMaxLoc(res)

            if max_val < 0.7:
                return None, f'{anchor["name"]} の信頼性が低すぎます ({max_val:.2f})。', None, None

            max_loc = (max_loc_relative[0] + roi_x, max_loc_relative[1] + roi_y)
            src_points.append([max_loc[0] + w / 2, max_loc[1] + h / 2])
            dst_points.append([ex + ew / 2, ey + eh / 2])
        except Exception as e:
            return None, f"アンカー処理中にエラー: {e}", None, None

    p1_src, p2_src = src_points
    p1_dst, p2_dst = dst_points
    dx_src, dy_src = p2_src[0] - p1_src[0], p2_src[1] - p1_src[1]
    dx_dst, dy_dst = p2_dst[0] - p1_dst[0], p2_dst[1] - p1_dst[1]
    p3_src = [p1_src[0] - dy_src, p1_src[1] + dx_src]
    p3_dst = [p1_dst[0] - dy_dst, p1_dst[1] + dx_dst]
    
    src_pts_np = np.float32([p1_src, p2_src, p3_src])
    dst_pts_np = np.float32([p1_dst, p2_dst, p3_dst])

    matrix = cv2.getAffineTransform(src_pts_np, dst_pts_np)
    return matrix, "成功", src_pts_np, dst_pts_np

# --- 解析実行 --- 
def run_analysis(config, image_dir, output_dir, output_csv, debug_dir):
    all_results = []
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))

    # config.jsonからアンカー画像のパスを取得し、解析対象から除外する
    if config.get("anchors"):
        anchor_template_paths = {os.path.abspath(a["template_path"]) for a in config.get("anchors", [])}
        image_paths = [p for p in image_paths if os.path.abspath(p) not in anchor_template_paths]

    if not image_paths: return print(f"⚠️ 解析対象の画像が {image_dir} に見つかりません。（アンカー画像は除外されています）")
    print(f"解析対象: {len(image_paths)}枚の画像")

    # デバッグモード用のデータ収集リストを初期化
    if DEBUG_MODE:
        ratios_by_checkbox = {cb['name']: [] for cb in config.get("checkboxes", [])}

    for path in tqdm(image_paths, desc="解析処理中", unit="枚"):
        img = cv2.imread(path)
        if img is None: continue

        corrected_img = img
        # get_affine_transformから座標情報も受け取る
        matrix, status, src_pts_np, dst_pts_np = get_affine_transform(img, config.get("anchors"))
        
        if matrix is not None:
            rows, cols, _ = img.shape
            corrected_img = cv2.warpAffine(img, matrix, (cols, rows))
        elif status != "アンカーが2つ定義されていません。":
             tqdm.write(f"  -> 警告: {os.path.basename(path)} の座標補正をスキップ。理由: {status}")

        debug_image = corrected_img.copy() if DEBUG_MODE else None
        result = {"file_path": os.path.basename(path)}

        # --- デバッグ: 補正の矢印と移動量を描画 ---
        if DEBUG_MODE and matrix is not None:
            # 元の画像のアンカー座標を、補正後の画像座標系に変換
            transformed_src_pts = cv2.transform(src_pts_np.reshape(-1, 1, 2), matrix)
            
            # 矢印と移動量テキストを描画 (アンカーの2点のみ)
            for i in range(2):
                anchor = config.get("anchors", [])[i]
                pt_src = transformed_src_pts[i][0]
                pt_dst = dst_pts_np[i]
                
                pt1 = (int(pt_src[0]), int(pt_src[1]))
                pt2 = (int(pt_dst[0]), int(pt_dst[1]))
                
                # 矢印を描画（マゼンタ色）
                cv2.arrowedLine(debug_image, pt1, pt2, (255, 0, 255), 2, tipLength=0.05)
                
                # 移動量を計算してテキストを描画
                dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
                text = f"{anchor['name']}: (dx={dx}, dy={dy})"
                ex, ey, _, _ = anchor["expected_bbox"]
                cv2.putText(debug_image, text, (ex, ey - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        gray_img = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)

        if DEBUG_MODE and config.get("anchors"):
            for anchor in config.get("anchors"):
                ex, ey, ew, eh = anchor["expected_bbox"]
                cv2.rectangle(debug_image, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                cv2.putText(debug_image, anchor["name"], (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        for cb in config.get("checkboxes", []):
            x, y, w, h = cb["bbox"]
            binary_roi = binary_img[y:y+h, x:x+w]
            checked, ratio = is_checked(binary_roi, cb["threshold"])
            result[cb["name"]] = int(checked)
            
            if DEBUG_MODE:
                ratios_by_checkbox[cb['name']].append(ratio)
                
                color = (0, 255, 0) if checked else (0, 0, 255)
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), color, 2)
                cv2.putText(debug_image, f'{ratio:.5f}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        any_text_found = False
        for ft in config.get("free_texts", []):
            fx, fy, fw, fh = ft["bbox"]
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

    # --- デバッグモード: ヒストグラム生成 ---
    if DEBUG_MODE:
        import matplotlib.pyplot as plt
        import japanize_matplotlib

        print("\n📊 デバッグモード: 黒ピクセル比率のヒストグラムを生成します...")
        has_any_data = any(ratios for ratios in ratios_by_checkbox.values())

        # --- 項目ごとのヒストグラム ---
        for name, ratios in ratios_by_checkbox.items():
            if not ratios:
                tqdm.write(f"  -> 項目「{name}」の比率データがないため、ヒストグラムをスキップします。")
                continue
            
            plt.figure(figsize=(12, 7))
            plt.hist(ratios, bins=100, alpha=0.75, label=f'全{len(ratios)}件のデータ')
            plt.title(f'項目「{name}」の黒ピクセル比率の分布', fontsize=16)
            plt.xlabel('黒ピクセル比率 (Black Pixel Ratio)', fontsize=12)
            plt.ylabel('出現回数 (Frequency)', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            mean_val = np.mean(ratios)
            median_val = np.median(ratios)
            plt.axvline(mean_val, color='r', linestyle='dashed', linewidth=1.5, label=f'平均値: {mean_val:.4f}')
            plt.axvline(median_val, color='g', linestyle='dashed', linewidth=1.5, label=f'中央値: {median_val:.4f}')
            plt.legend()
            plt.tight_layout()
            hist_path = os.path.join(debug_dir, f'ratio_histogram_{name}.png')
            plt.savefig(hist_path)
            plt.close()
        
        # --- 全項目を重ねたヒストグラム ---
        if has_any_data:
            print("📊 デバッグモード: 全項目を重ね合わせたヒストグラムを生成します...")
            plt.figure(figsize=(12, 7))
            plt.title('全項目の黒ピクセル比率の分布（重ね合わせ）', fontsize=16)
            plt.xlabel('黒ピクセル比率 (Black Pixel Ratio)', fontsize=12)
            plt.ylabel('出現回数 (Frequency)', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            for name, ratios in ratios_by_checkbox.items():
                if not ratios:
                    continue
                plt.hist(ratios, bins=100, alpha=0.6, label=name)

            plt.legend(title='質問項目')
            plt.tight_layout()
            combined_hist_path = os.path.join(debug_dir, 'ratio_histogram_ALL_COMBINED.png')
            plt.savefig(combined_hist_path)
            plt.close()
        
        print(f"✅ ヒストグラムが {debug_dir} に保存されました。")

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ 全画像解析完了。結果を {output_csv} に出力しました。")
    if DEBUG_MODE: print(f"✅ デバッグ画像が {debug_dir} に保存されました。")

# --- メイン処理フロー ---
if __name__ == "__main__":
    print("--- アンケート解析スクリプト ---")
    run_analysis(CONFIG, IMAGE_DIR, OUTPUT_DIR, OUTPUT_CSV, DEBUG_DIR)
