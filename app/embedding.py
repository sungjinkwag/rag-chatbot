# app/embedding.py

import os
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

# .env 환경변수 로드
load_dotenv()

# OpenAI API 키 불러오기
openai_api_key = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 생성
client = OpenAI(api_key=openai_api_key)

# 임베딩 함수 정의
def get_embedding(text: str, model: str = "text-embedding-ada-002") -> list[float]:
    try:
        response = client.embeddings.create(input=[text],model=model)
        return response.data[0].embedding
    except RateLimitError as e:
        print("❌ OpenAI Rate Limit 초과 또는 결제 필요:", e)
        raise e
