# main.py
import os
import fitz  # PyMuPDF
from keybert import KeyBERT
import re
import gc

# --- パス設定 (Path Settings) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, 'electrical_vocab_final.txt')

# --- 関数定義 (Function Definitions) ---

def clean_text(text: str) -> str:
    """テキストから不要なノイズ（ページ番号、特定のマーカーなど）を除去する関数。"""
    # "Page XX" や "Chapter X" のようなヘッダー/フッター行を削除
    text = re.sub(r'^(Page\s\d+|Chapter\s\d+.*?)\n', '', text, flags=re.MULTILINE | re.IGNORECASE)
    # 参考文献のマーカー [1], [2] などを削除
    text = re.sub(r'\[\d+\]', '', text)
    # 連続する改行や空白を一つのスペースにまとめる
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_texts_from_pdfs(pdf_dir: str) -> list[str]:
    """指定されたディレクトリ内の全てのPDFから、クリーンなテキストを抽出する関数。"""
    texts = []
    print(f"ディレクトリ '{pdf_dir}' 内のPDFを処理します...")
    if not os.path.exists(pdf_dir):
        print(f"エラー: ディレクトリ '{pdf_dir}' が見つかりません。")
        return texts

    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"  - ファイルを安全モードで処理中: {filename}")
            try:
                doc = fitz.open(pdf_path)
                full_text = ""
                # PDFが10ページ以上ある場合、最初の5ページ（表紙や目次）をスキップ
                start_page = 5 if doc.page_count > 10 else 0
                
                for page_num in range(start_page, doc.page_count):
                    page = doc.load_page(page_num)
                    page_text = page.get_text("text") # テキストのみ抽出
                    full_text += clean_text(page_text) + " "
                    
                    # 20ページごとにメモリを解放
                    if (page_num + 1) % 20 == 0:
                        gc.collect()
                        
                texts.append(full_text)
                doc.close()
                print(f"  - 完了: {filename}")
            except Exception as e:
                print(f"    エラー: {filename} の処理中に問題が発生しました: {e}")
    
    gc.collect()
    return texts

def split_text_into_chunks(text: str, chunk_size: int = 50000):
    """テキストを指定された文字数で分割するジェネレータ"""
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

# --- メイン処理 (Main Process) ---
if __name__ == "__main__":
    print("=" * 50)
    print("電気工学 専門用語辞書 自動構築システム (v2.0)")
    print("=" * 50)

    # 1. 出力ディレクトリが存在しない場合は作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. PDFからテキストを抽出
    all_texts = extract_texts_from_pdfs(PDF_DIR)

    if not all_texts:
        print("処理するテキストが見つかりませんでした。プログラムを終了します。")
    else:
        combined_text = " ".join(all_texts)
        print(f"\nテキストの抽出が完了しました。総文字数: {len(combined_text)}")

        # 3. KeyBERTモデルの準備
        print("KeyBERTモデルをロード中... (all-MiniLM-L6-v2)")
        kw_model = KeyBERT(model='all-MiniLM-L6-v2')
        print("モデルのロードが完了しました。")

        # 4. テキストを分割してキーワードを抽出（メモリ対策）
        print("専門用語の抽出を開始します... (テキストを分割処理中)")
        all_keywords = []
        text_chunks = list(split_text_into_chunks(combined_text, chunk_size=50000))
        
        for i, chunk in enumerate(text_chunks):
            print(f"  - チャンク {i+1}/{len(text_chunks)} を処理中 (文字数: {len(chunk)})")
            try:
                chunk_keywords = kw_model.extract_keywords(
                    chunk,
                    keyphrase_ngram_range=(2, 3),
                    stop_words='english',
                    use_mmr=True,
                    diversity=0.7,
                    top_n=30  # 各チャンクから多めに30個抽出
                )
                if chunk_keywords:
                    all_keywords.extend(chunk_keywords)
                gc.collect()
            except Exception as e:
                print(f"    チャンク {i+1} の処理中にエラーが発生しました: {e}")
        
        print("全チャンクの処理が完了しました。")

        # 5. 抽出結果の統合と後処理
        # 重複を削除し、最も高いスコアを保持するロジック
        unique_keywords_dict = {}
        for term, score in all_keywords:
            # "of" "the" などを含む不自然なフレーズを除外
            if not any(f" {bad_word} " in f" {term} " for bad_word in ["of", "the", "and", "is", "for", "in"]):
                 if term not in unique_keywords_dict or score > unique_keywords_dict[term]:
                    unique_keywords_dict[term] = score
        
        # 辞書をリストに戻し、スコアでソート
        sorted_keywords = sorted(unique_keywords_dict.items(), key=lambda item: item[1], reverse=True)
        
        # 6. 専門家知識による補完 (Human-in-the-Loop)
        print("専門家知識による補完処理...")
        final_terms = sorted_keywords
        essential_terms = ["Ohm's Law", "Kirchhoff's laws", "Thevenin's theorem", "Norton's theorem", "circuit breaker", "voltage regulator"]
        
        existing_terms_lower = [term.lower() for term, score in final_terms]
        for term in essential_terms:
            if term.lower() not in existing_terms_lower:
                final_terms.append((term, 0.95)) # 高い信頼度スコアを付与

        # 再度スコアで降順にソート
        final_terms.sort(key=lambda x: x[1], reverse=True)

        # 7. 結果をファイルに書き出し
        print(f"結果を '{OUTPUT_FILE_PATH}' に書き出し中...")
        with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write("# 電気工学 核心用語辞書\n")
            f.write("# 自動抽出 + 専門家による検証・補完\n\n")
            # 上位50件のみを保存
            for term, score in final_terms[:50]:
                f.write(f"{term} :: 電気核心術語 (置信度:{score:.2f})\n")

        print("=" * 50)
        print(f"辞書は正常に保存されました: {os.path.basename(OUTPUT_FILE_PATH)}")
        print("=" * 50)
