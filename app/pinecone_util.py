# app/pinecone_util.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone

# 환경변수 로드
load_dotenv()

# API 키 및 인덱스 정보 불러오기
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")

# Pinecone 클라이언트 생성
pc = Pinecone(api_key=pinecone_api_key)

# 인덱스 객체
index = pc.Index(pinecone_index)

# 벡터 삽입
def upsert_vector(id: str, vector: list[float], metadata: dict = {}):
    index.upsert(vectors=[(id, vector, metadata)])

# 벡터 검색
def query_vector(vector: list[float], top_k: int = 3):
    return index.query(vector=vector, top_k=top_k, include_metadata=True)
