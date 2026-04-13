"""
llm.py
======
FREE LLM answers using Groq API (llama-3.3-70b-versatile).
Get your free key at: https://console.groq.com
"""

from groq import Groq
from typing import List

CHAT_MODEL = "llama-3.3-70b-versatile"  # fast, free on Groq

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions about PDF documents.
You will be given relevant excerpts from the document as context.

Instructions:
- Answer ONLY based on the provided context.
- If the context does not contain enough information, say so clearly.
- Be concise but thorough.
- Do not make up information that is not in the context.
"""


def generate_answer(
    question: str,
    context_chunks: List[str],
    api_key: str,
    max_tokens: int = 600,
) -> str:
    """
    Generate an answer using Groq's free LLM API (RAG step).

    Args:
        question:       The user's question.
        context_chunks: Retrieved text chunks from Endee.
        api_key:        Your Groq API key (gsk_...).
        max_tokens:     Max tokens in the response.

    Returns:
        The answer as a string.
    """
    client = Groq(api_key=api_key)

    # Build context block from retrieved chunks
    context_text = "\n\n".join(
        f"[Context {i+1}]:\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    user_message = f"""Context from the document:
{context_text}

---
Question: {question}

Please answer based on the context above."""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    return answer.strip() if answer else "Sorry, I could not generate an answer."
