# main_v1.py
import os
import fitz  # PyMuPDF
from keybert import KeyBERT

# --- パス設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, 'electrical_vocab_v1_naive.txt') # 結果ファイル名を変更

# --- メイン処理 ---
if __name__ == "__main__":
    print("--- 専門用語辞書 自動構築システム (v1.0 - 初期アプローチ) ---")

    # 1. 出力ディレクトリが存在しない場合は作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. PDFからテキストを抽出（単純な方法）
    texts = []
    print(f"'{PDF_DIR}' ディレクトリ内のPDFを処理します...")
    if not os.path.exists(PDF_DIR):
        print(f"エラー: '{PDF_DIR}' ディレクトリが見つかりません。")
    else:
        for filename in os.listdir(PDF_DIR):
            if filename.lower().endswith(".pdf"):
                print(f"  - ファイルを処理中: {filename}")
                try:
                    doc = fitz.open(os.path.join(PDF_DIR, filename))
                    full_text = ""
                    for page in doc:
                        # 改行をスペースに置換するだけの単純なクリーニング
                        full_text += page.get_text().replace('\n', ' ')
                    texts.append(full_text)
                    doc.close()
                except Exception as e:
                    print(f"    エラー: {filename} の処理中に問題が発生しました: {e}")

    if not texts:
         print("処理するテキストが見つかりませんでした。プログラムを終了します。")
    else:
        combined_text = " ".join(texts)
        print(f"\nテキストの抽出が完了しました。総文字数: {len(combined_text)}")
        
        # 3. KeyBERTモデルの準備（初期版）
        print("KeyBERTモデルをロード中... (paraphrase-multilingual-MiniLM-L12-v2)")
        # 初期に検討した、より汎用的で重いモデル
        kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
        print("モデルのロードが完了しました。")

        # 4. キーワード抽出（初期版のパラメータ）
        print("専門用語の抽出を開始します...")
        keywords = kw_model.extract_keywords(
            combined_text,
            keyphrase_ngram_range=(1, 3),    # 単独の単語も許容してしまう
            stop_words=["this", "is"],       # 限定的なストップワードのみ
            top_n=50
            # 多様性の確保（MMR）などを行わない
        )
        print("専門用語の抽出が完了しました。")

        # 5. 結果をファイルに書き出し（後処理なし）
        print(f"結果を '{OUTPUT_FILE_PATH}' に書き出し中...")
        with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            for term, score in keywords:
                f.write(f"{term} :: 抽出された用語 (スコア:{score:.2f})\n")

        print("\n処理が正常に完了しました！")