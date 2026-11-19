import re
from functools import lru_cache

from fastapi import Body, FastAPI, HTTPException
from transformers import pipeline

MODELO = "Helsinki-NLP/opus-mt-en-fr"

app = FastAPI(title="TP2 - Parte 1 - Questao 2 - Traducao EN-FR com Helsinki-NLP")


@lru_cache(maxsize=None)
def get_translator():
    return pipeline("translation", model=MODELO)


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part for part in parts if part]


def translate(text):
    return get_translator()(text, max_length=512)[0]["translation_text"]


def translate_long(text):
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return translate(text)
    return " ".join(translate(sentence) for sentence in sentences)


@app.get("/")
def root():
    return {"modelo": MODELO, "endpoint": "/translate", "metodo": "POST"}


@app.post("/translate")
def traduzir_en_fr(text: str = Body(..., embed=True, max_length=10000)):
    if not text.strip():
        raise HTTPException(status_code=400, detail="O campo text nao pode ser vazio.")
    try:
        traducao = translate_long(text)
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Falha na traducao: {erro}")
    return {
        "modelo": MODELO,
        "origem": "en",
        "destino": "fr",
        "sentencas": len(split_sentences(text)),
        "texto_original": text,
        "texto_traduzido": traducao,
    }
