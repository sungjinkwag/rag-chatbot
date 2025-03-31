# app/gpt.py

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# GPT 응답 생성 함수
def generate_answer(question: str, contexts: list[str], sources: list[str], model: str = "gpt-3.5-turbo") -> str:
    context_text = "\n\n".join(
        [f"{i+1}. {ctx}" for i, ctx in enumerate(contexts)]
    )
    source_text = "\n".join(
        [f"[{i+1}] {url}" for i, url in enumerate(sources)]
    )

    system_prompt = (
        "당신은 친절한 기술 문서 요약 챗봇입니다. 아래 문서 내용을 참고하여 사용자의 질문에 친절하고 정확하게 답변하고, "
        "마지막에 참조 링크들을 출력해주세요."
    )

    user_prompt = f"""다음은 참고 문서입니다:

{context_text}

참고 문서 링크:
{source_text}

질문: {question}
답변:"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()

