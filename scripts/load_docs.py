# scripts/load_docs.py

import requests
from bs4 import BeautifulSoup
from app.embedding import get_embedding
from app.pinecone_util import upsert_vector
import time
import re
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BASE_URL = "https://react.dev"
START_PATH = "/learn"

# ✅ 전체 링크 자동 수집 함수
def collect_internal_links(start_path: str) -> list[str]:
    visited = set()
    to_visit = [start_path]

    while to_visit:
        path = to_visit.pop()
        if path in visited:
            continue
        visited.add(path)

        full_url = f"{BASE_URL}{path}"
        try:
            res = requests.get(full_url)
            if res.status_code != 200:
                continue
            soup = BeautifulSoup(res.text, "html.parser")
            for link in soup.find_all("a", href=True):
                href = link["href"]
                # /learn/...으로 시작하는 내부 문서 링크만
                if href.startswith("/learn") and href not in visited:
                    to_visit.append(href)
        except:
            continue

    return list(visited)

# ✅ 본문 정제
def clean_text(html):
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup
    for tag in main(["nav", "footer", "header", "script", "style"]):
        tag.decompose()
    return main.get_text(separator="\n").strip()

# ✅ 텍스트 분할
def split_into_chunks(text, max_chars=800):
    chunks, chunk = [], ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(chunk) + len(line) > max_chars:
            chunks.append(chunk.strip())
            chunk = line
        else:
            chunk += " " + line
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# ✅ 크롤링 및 Pinecone 저장 전체 수행
def crawl_and_store_all():
    all_paths = collect_internal_links(START_PATH)
    print(f"🔗 총 {len(all_paths)}개의 페이지 수집됨")

    for path in all_paths:
        url = f"{BASE_URL}{path}"
        print(f"...크롤링 중: {url}")
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"❌ 요청 실패: {url}")
                continue

            text = clean_text(response.text)
            chunks = split_into_chunks(text)

            for i, chunk in enumerate(chunks):
                embedding = get_embedding(chunk)
                doc_id = f"{path.replace('/', '_')}_{i}"
                upsert_vector(doc_id, embedding, {
                    "text": chunk,
                    "source": url
                })
                print(f"✅ 저장 완료: {doc_id}")
                time.sleep(0.3)  # OpenAI API 제한 대비
        except Exception as e:
            print(f"❗ 에러 발생 ({url}): {e}")

if __name__ == "__main__":
    crawl_and_store_all()
