# app/main.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from app.embedding import get_embedding
from app.pinecone_util import upsert_vector, query_vector
from app.gpt import generate_answer
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# ✅ CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",  # Local 개발 서버 주소
                   "https://react-chatbot-frontend.vercel.app"],  # React Frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# .env 환경변수 로딩
load_dotenv()

# 환경변수 확인용
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# 요청 바디 정의 (Pydantic 사용)
class ChatRequest(BaseModel):
    query: str

# 테스트용 루트 라우터
@app.get("/")
def root():
    return {"message": "Hello from RAG Chatbot!"}

@app.post("/test-save")
def test_save():
    text = "React는 UI를 구축하기 위한 라이브러리입니다."
    vector = get_embedding(text)
    upsert_vector("doc1", vector, {"text": text})
    return {"message": "벡터 저장 완료"}

@app.post("/test-search")
def test_search(request: ChatRequest):
    vector = get_embedding(request.query)
    result = query_vector(vector)
    return result.to_dict()

# 질문을 받는 엔드포인트
@app.post("/chat")
def chat(request: ChatRequest):
    # 1. 질문 임베딩
    vector = get_embedding(request.query)

    # 2. Pinecone에서 유사 문서 검색
    result = query_vector(vector)

    # 3. 텍스트와 source URL 따로 추출
    texts = []
    sources = []
    for match in result["matches"]:
        metadata = match["metadata"]
        texts.append(metadata["text"])
        sources.append(metadata.get("source", "출처 없음"))

    # 4. GPT 응답 생성 (텍스트만 추출해 전달)
    answer = generate_answer(request.query, texts, sources)

    return {
        #"question": request.query,
        "answer": answer,
        "references": [
            {"source": src, "text": txt}
            for txt, src in zip(texts, sources)
        ]
    }

@app.post("/embedding")
def embed_test(request: ChatRequest):
    embedding = get_embedding(request.query)
    return {
        "length": len(embedding),
        "sample": embedding[:5]  # 앞 5개만 출력
    }