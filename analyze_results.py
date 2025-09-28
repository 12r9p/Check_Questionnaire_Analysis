import pandas as pd
import itertools
import os

def analyze_survey_results(csv_path="results.csv", output_path="analysis_summary.md"):
    """
    results.csvを読み込み、統計的なサマリーを生成してテキストファイルに出力する。
    """
    # --- 1. データの読み込み ---
    if not os.path.exists(csv_path):
        print(f"エラー: {csv_path} が見つかりません。先にmain.pyを実行して解析を行ってください。")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"エラー: {csv_path} の読み込みに失敗しました。ファイルが破損している可能性があります。: {e}")
        return

    report_lines = []

    # --- 2. 基本統計 ---
    total_responses = len(df)
    report_lines.append("# アンケート結果分析レポート")
    report_lines.append("\n## 基本統計")
    report_lines.append(f"- 総回答数: {total_responses}件")

    # 解析対象の列を特定 (file_pathとコピー先列を除外)
    item_columns = [col for col in df.columns if col not in ["file_path", "free_text_image_copied"]]

    # --- 3. 単純集計 ---
    report_lines.append("\n## 各項目の単純集計")
    report_lines.append("各項目が選択された数と、全体に占める割合を示します。\n")

    for col in item_columns:
        count = df[col].sum()
        percentage = (count / total_responses) * 100 if total_responses > 0 else 0
        report_lines.append(f"- **{col}**: {count}票 ({percentage:.1f}%)")

    # --- 4. クロス集計 ---
    report_lines.append("\n---")
    report_lines.append("\n## クロス集計")
    report_lines.append("2つの項目間の関連性を示します。「Aを選んだ人はBも選んでいるか」などを分析できます。\n")

    # 列のペアを作成
    column_pairs = list(itertools.combinations(item_columns, 2))

    for pair in column_pairs:
        item1, item2 = pair
        
        report_lines.append(f"### **{item1}** と **{item2}** の関係")
        
        # クロス集計表を生成
        crosstab = pd.crosstab(df[item1], df[item2])
        
        # Markdownテーブル形式に変換
        report_lines.append(crosstab.to_markdown())
        report_lines.append("\n")

    # --- 5. レポートをファイルに出力 ---
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"✅ 分析レポートが {output_path} に保存されました。")
    except Exception as e:
        print(f"エラー: レポートファイルの書き込みに失敗しました。: {e}")

if __name__ == "__main__":
    # pandasの表示オプションを設定（念のため）
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    
    analyze_survey_results()

