# main.py
import os
import fitz  # PyMuPDF
from keybert import KeyBERT
import re
import gc

# --- パス設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, 'electrical_vocab_final.txt')

# --- 関数定義 ---

def clean_text(text: str) -> str:
    """
    テキストから不要なノイズ（ページ番号、特定のマーカーなど）を除去する関数。
    正規表現を使用。
    """
    # "Page XX" や "Chapter X" のようなヘッダー/フッター行を削除
    text = re.sub(r'^(Page\s\d+|Chapter\s\d+.*?)\n', '', text, flags=re.MULTILINE | re.IGNORECASE)
    # 参考文献のマーカー [1], [2] などを削除
    text = re.sub(r'\[\d+\]', '', text)
    # 連続する改行や空白を一つのスペースにまとめる
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_texts_from_pdfs(pdf_dir: str) -> list[str]:
    """
    指定されたディレクトリ内の全てのPDFから、クリーンなテキストを抽出する関数。
    """
    texts = []
    print(f"'{pdf_dir}' ディレクトリ内のPDFを処理します...")
    if not os.path.exists(pdf_dir):
        print(f"エラー: '{pdf_dir}' ディレクトリが見つかりません。")
        return texts

    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"  - ファイルを処理中: {filename}")
            try:
                doc = fitz.open(pdf_path)
                full_text = ""
                for page_num, page in enumerate(doc):
                    page_text = page.get_text()
                    full_text += clean_text(page_text) + " "
                    # メモリを定期的に解放
                    if (page_num + 1) % 20 == 0:
                        gc.collect()
                texts.append(full_text)
                doc.close()
            except Exception as e:
                print(f"    エラー: {filename} の処理中に問題が発生しました: {e}")
    
    # メモリ解放
    gc.collect()
    return texts

# --- メイン処理 ---

if __name__ == "__main__":
    print("--- 専門用語辞書 自動構築システム (v2.0) ---")

    # 1. 出力ディレクトリが存在しない場合は作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. PDFからテキストを抽出
    all_texts = extract_texts_from_pdfs(PDF_DIR)

    if not all_texts:
        print("処理するテキストが見つかりませんでした。プログラムを終了します。")
    else:
        combined_text = " ".join(all_texts)
        print(f"\nテキストの抽出が完了しました。総文字数: {len(combined_text)}")

        # 3. KeyBERTモデルの準備（改良版）
        print("KeyBERTモデルをロード中... (all-MiniLM-L6-v2)")
        # より軽量で高性能な英語特化モデルを使用
        kw_model = KeyBERT(model='all-MiniLM-L6-v2')
        print("モデルのロードが完了しました。")

        # 4. キーワード抽出（改良版のパラメータ）
        print("専門用語の抽出を開始します...")
        keywords = kw_model.extract_keywords(
            combined_text,
            keyphrase_ngram_range=(2, 3),      # 意味のある複合語（2-3語）のみを対象
            stop_words='english',              # 英語の一般的なストップワードを除外
            use_mmr=True,                      # 多様性を確保し、似たような単語の重複を防ぐ (MMR)
            diversity=0.7,                     # 多様性の度合い (0.7は高め)
            top_n=50                           # 上位50件を抽出
        )
        print("専門用語の抽出が完了しました。")

        # 5. AIが見逃した基礎用語を専門家が補完 (Human-in-the-Loop)
        print("専門家知識による補完処理...")
        final_terms = keywords
        essential_terms = ["Ohm's Law", "Kirchhoff's laws", "Thevenin's theorem", "Norton's theorem"]
        
        # 辞書にまだない基礎用語を追加
        existing_terms_lower = [term.lower() for term, score in final_terms]
        for term in essential_terms:
            if term.lower() not in existing_terms_lower:
                # 人手で追加した用語には高い信頼度スコアを付与
                final_terms.append((term, 0.95))

        # スコアで降順にソート
        final_terms.sort(key=lambda x: x[1], reverse=True)


        # 6. 結果をファイルに書き出し
        print(f"結果を '{OUTPUT_FILE_PATH}' に書き出し中...")
        with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            for term, score in final_terms:
                f.write(f"{term} :: 電気核心術語 (置信度:{score:.2f})\n")

        print("\n処理が正常に完了しました！")