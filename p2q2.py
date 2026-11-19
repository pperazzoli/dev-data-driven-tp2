import os
from functools import lru_cache

from fastapi import Body, FastAPI, HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

app = FastAPI(title="TP2 - Parte 2 - Questao 2 - Traducao EN-FR com LangChain e OpenAI")


@lru_cache(maxsize=None)
def get_chat_model():
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0)


def translate_with_chat_model(text, target_language="French"):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional translator. Translate the user text into {target_language}. Answer only with the translation, keeping the original formatting."),
        ("human", "{text}"),
    ])
    chain = prompt | get_chat_model() | StrOutputParser()
    return chain.invoke({"text": text, "target_language": target_language}).strip()


@app.get("/")
def root():
    return {"modelo": OPENAI_MODEL, "endpoint": "/translate", "metodo": "POST"}


@app.post("/translate")
def traduzir_com_openai(
    text: str = Body(..., embed=True, max_length=10000),
    target_language: str = Body("French", embed=True, max_length=40),
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="O campo text nao pode ser vazio.")
    try:
        traducao = translate_with_chat_model(text, target_language=target_language)
    except Exception as erro:
        raise HTTPException(status_code=502, detail=f"Falha na chamada a OpenAI: {erro}")
    return {
        "modelo": OPENAI_MODEL,
        "origem": "en",
        "destino": target_language,
        "texto_original": text,
        "texto_traduzido": traducao,
    }
