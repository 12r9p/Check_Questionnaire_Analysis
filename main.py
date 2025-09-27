# --- ライブラリのインポート ---
import os
import shutil
import glob
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Colab環境でのみ動作
try:
    from google.colab import files
except ImportError:
    files = None

# --- 日本語フォント設定 ---
# Matplotlibで日本語が文字化けするのを防ぐ
try:
    import japanize_matplotlib
except ImportError:
    import subprocess
    import sys
    print("日本語表示用のライブラリ 'japanize-matplotlib' をインストールします。")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "japanize-matplotlib"])
    import japanize_matplotlib

# --- 初期設定 ---
# 画像を保存するディレクトリ
IMAGE_DIR = "./images"
# 自由記述があった画像を出力するディレクトリ
OUTPUT_DIR = "./free_texts"
# 出力CSVファイル名
OUTPUT_CSV = "results.csv"

# ディレクトリ作成
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- アンケート定義 ---
# bbox: [x, y, width, height]
CONFIG = {
    "checkboxes": [
        {"name": "Q1_選択肢1", "bbox": [100, 200, 20, 20], "threshold": 0.1},
        {"name": "Q1_選択肢2", "bbox": [200, 200, 20, 20], "threshold": 0.1},
        {"name": "Q2_選択肢1", "bbox": [100, 300, 20, 20], "threshold": 0.1},
        {"name": "Q2_選択肢2", "bbox": [200, 300, 20, 20], "threshold": 0.1},
        # 必要に応じて追加
    ],
    "free_texts": [
        {"name": "感想", "bbox": [100, 400, 500, 100], "threshold": 0.01},
        {"name": "その他", "bbox": [100, 550, 500, 100], "threshold": 0.01},
        # 必要に応じて追加
    ]
}
print("✅ CONFIG読み込み完了")


# --- 複数画像をアップロード ---
# Colab環境でローカルから画像をアップロード
def upload_images_to_colab(target_dir):
    """Colab環境で画像をアップロードし、指定ディレクトリに保存する"""
    if files is None:
        print("（Colab環境ではないため、アップロード機能はスキップされました）")
        # ローカル環境で手動で target_dir に画像を入れてください
        return None

    print(f"{target_dir} に画像をアップロードしてください。")
    uploaded = files.upload()
    if not uploaded:
        print("画像がアップロードされませんでした。")
        return None

    for name, data in uploaded.items():
        with open(os.path.join(target_dir, name), "wb") as f:
            f.write(data)
    print(f"✅ {len(uploaded)}枚の画像を {target_dir} に保存しました")
    return list(uploaded.keys())[0] # サンプル画像ファイル名を返す


# --- 座標確認用のグリッド表示 ---
def show_image_with_grid(image_path, grid_spacing=200, sub_grid_spacing=50):
    """
    画像に座標確認用のグリッドを重ねて表示する。
    インタラクティブ機能が使えない環境向けの代替案。
    """
    if not os.path.exists(image_path):
        print(f"エラー: 画像ファイルが見つかりません: {image_path}")
        return
    image = cv2.imread(image_path)
    if image is None:
        print(f"エラー: 画像を読み込めません: {image_path}")
        return

    height, width, _ = image.shape

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # グリッドの設定
    ax.set_xticks(np.arange(0, width + 1, grid_spacing))
    ax.set_yticks(np.arange(0, height + 1, grid_spacing))
    ax.set_xticks(np.arange(0, width + 1, sub_grid_spacing), minor=True)
    ax.set_yticks(np.arange(0, height + 1, sub_grid_spacing), minor=True)

    # グリッドの描画スタイル
    ax.grid(which='major', linestyle='-', linewidth='1', color='red')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='blue')
    
    # 軸の目盛りを画像の左と上に表示
    ax.xaxis.tick_top()
    
    ax.set_title("グリッド付き画像 (座標を手動で確認してください)")
    plt.show()

    print("\n--- 座標の手動設定ガイド ---")
    print("表示された画像のグリッドを参考に、CONFIG内のbbox座標 [x, y, w, h] を手動で更新してください。")
    print(f"主グリッド(赤線)の間隔: {grid_spacing}px")
    print(f"補助グリッド(青点線)の間隔: {sub_grid_spacing}px")
    print("bboxは [左上のx座標, 左上のy座標, 幅, 高さ] の順です。")


# --- 設定の可視化 ---
# CONFIGで指定した領域が正しいか、サンプル画像上で確認します。
def visualize_config(image_path, config):
    """設定されたチェックボックスと自由記述欄を画像上で可視化する"""
    if not os.path.exists(image_path):
        print(f"エラー: 画像ファイルが見つかりません: {image_path}")
        return
    image = cv2.imread(image_path)
    if image is None:
        print(f"エラー: 画像を読み込めません: {image_path}")
        return

    preview = image.copy()

    # チェックボックスの矩形を青で描画
    for cb in config["checkboxes"]:
        x, y, w, h = cb["bbox"]
        cv2.rectangle(preview, (x, y), (x+w, y+h), (255, 0, 0), 2)  # 青 (BGR)

    # 自由記述欄を緑で描画
    for ft in config["free_texts"]:
        fx, fy, fw, fh = ft["bbox"]
        cv2.rectangle(preview, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)  # 緑 (BGR)

    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("チェックボックスと自由記述欄のプレビュー")
    plt.show()

    # 座標表示
    print("チェックボックス座標一覧:", [cb["bbox"] for cb in config["checkboxes"]])
    print("自由記述欄座標一覧:", [ft["bbox"] for ft in config["free_texts"]])


# --- 解析用関数 ---
def get_black_pixel_ratio(roi):
    """ROI内の黒ピクセルの割合を計算する（適応的しきい値処理を使用）"""
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 適応的しきい値処理で、照明ムラに強い二値化を行う
    block_size = 11  # 小領域のサイズ（奇数）
    C = 5          # 平均から引く定数
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C
    )
    
    black_pixels = np.count_nonzero(binary)
    total_pixels = roi.shape[0] * roi.shape[1]
    return black_pixels / total_pixels

def is_checked(roi, threshold):
    """チェックボックスがチェックされているか判定する"""
    ratio = get_black_pixel_ratio(roi)
    return ratio >= threshold, ratio

def has_text(roi, threshold):
    """自由記述欄にテキストがあるか判定する"""
    ratio = get_black_pixel_ratio(roi)
    return ratio >= threshold


# --- 解析実行 ---
def run_analysis(config, image_dir, output_dir, output_csv):
    """指定されたディレクトリの画像を解析し、結果をCSVに出力する"""
    all_results = []
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(image_dir, "*.png")) + \
                  glob.glob(os.path.join(image_dir, "*.jpeg"))

    if not image_paths:
        print(f"⚠️ 解析対象の画像が {image_dir} に見つかりません。")
        return

    print(f"解析対象: {len(image_paths)}枚の画像")

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ 読み込み失敗: {path}")
            continue

        result = {"file_path": os.path.basename(path)}
        console_line = os.path.basename(path)

        # 各チェックボックス判定
        for cb in config["checkboxes"]:
            x, y, w, h = cb["bbox"]
            roi = img[y:y+h, x:x+w]
            checked, _ = is_checked(roi, cb["threshold"])
            result[cb["name"]] = int(checked)
            console_line += f" {cb['name']}:{'1' if checked else '0'}"

        # 各自由記述判定
        any_text_found = False
        for ft in config["free_texts"]:
            fx, fy, fw, fh = ft["bbox"]
            free_text_roi = img[fy:fy+fh, fx:fx+fw]
            text_found = has_text(free_text_roi, ft["threshold"])
            result[ft["name"]] = int(text_found)
            console_line += f" {ft['name']}:{'1' if text_found else '0'}"
            if text_found:
                any_text_found = True

        # いずれかの自由記述があれば画像をコピー
        if any_text_found:
            basename = os.path.basename(path)
            copy_path = os.path.join(output_dir, basename)
            shutil.copy(path, copy_path)
            result["free_text_image_copied"] = os.path.basename(copy_path)
        else:
            result["free_text_image_copied"] = ""

        all_results.append(result)
        print(console_line)

    # CSV出力
    if not all_results:
        print("解析結果がありません。")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ 全画像解析完了。結果を {output_csv} に出力しました。")
    print(f"✅ 自由記述があった画像は {output_dir} にコピーされました。")

    # Colab環境の場合、CSVをダウンロード
    if files:
        try:
            files.download(output_csv)
            print(f"✅ {output_csv} のダウンロードを開始します。")
        except Exception as e:
            print(f"ダウンロード中にエラーが発生しました: {e}")

# --- デバッグ用セル ---
def debug_and_visualize_one_by_one(config, image_dir):
    """
    画像一枚ごとに解析結果を可視化して表示するデバッグ用関数。
    チェックあり・テキストありは緑、なしは赤の枠で表示される。
    """
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(image_dir, "*.png")) + \
                  glob.glob(os.path.join(image_dir, "*.jpeg"))

    if not image_paths:
        print(f"⚠️ デバッグ対象の画像が {image_dir} に見つかりません。")
        return

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ 読み込み失敗: {path}")
            continue

        preview = img.copy()
        print(f"--- デバッグ表示: {os.path.basename(path)} ---")

        # 各チェックボックス判定と描画
        for cb in config["checkboxes"]:
            x, y, w, h = cb["bbox"]
            roi = img[y:y+h, x:x+w]
            checked, ratio = is_checked(roi, cb["threshold"])
            color = (0, 255, 0) if checked else (0, 0, 255) # BGR
            cv2.rectangle(preview, (x, y), (x+w, y+h), color, 2)
            print(f"  - {cb['name']}: {'チェックあり' if checked else 'なし'} (黒ピクセル率: {ratio:.3f})")

        # 各自由記述判定と描画
        for ft in config["free_texts"]:
            fx, fy, fw, fh = ft["bbox"]
            free_text_roi = img[fy:fy+fh, fx:fx+fw]
            text_found = has_text(free_text_roi, ft["threshold"])
            color = (0, 255, 0) if text_found else (0, 0, 255) # BGR
            cv2.rectangle(preview, (fx, fy), (fx+fw, fy+fh), color, 2)
            print(f"  - {ft['name']}: {'あり' if text_found else 'なし'}")

        # 画像表示
        plt.figure(figsize=(12, 12))
        plt.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
        plt.title(f"デバッグ表示: {os.path.basename(path)}")
        plt.axis("off")
        plt.show()


# --- メイン処理フロー ---
# 以下の①〜⑤のコメントを一つずつ外して、目的に合わせて実行してください。

# ① 画像をアップロード
# sample_image_filename = upload_images_to_colab(IMAGE_DIR)

# ② (初回設定時のみ) 座標を確認するため、グリッド付きで画像を表示する
# 表示されたグリッドを参考に、手動でCONFIGの座標を編集してください。
# if 'sample_image_filename' in locals() and sample_image_filename:
#     sample_image_path = os.path.join(IMAGE_DIR, sample_image_filename)
#     show_image_with_grid(sample_image_path)
# else:
#     print("画像がないため、グリッド表示をスキップします。手動で " + IMAGE_DIR + " に画像を入れてください。")

# ③ (設定後) 設定が正しいかプレビューで確認
# if 'sample_image_filename' in locals() and sample_image_filename:
#     sample_image_path = os.path.join(IMAGE_DIR, sample_image_filename)
#     visualize_config(sample_image_path, CONFIG)
# else:
#     print("画像がないため、設定の可視化をスキップします。")

# ④ (デバッグ用) 解析結果を一枚ずつ見て確認
# debug_and_visualize_one_by_one(CONFIG, IMAGE_DIR)

# ⑤ (本番) 全画像を解析してCSV出力
# run_analysis(CONFIG, IMAGE_DIR, OUTPUT_DIR, OUTPUT_CSV)