import re
from functools import lru_cache

from fastapi import Body, FastAPI, HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

MODELO = "Helsinki-NLP/opus-mt-en-de"

app = FastAPI(title="TP2 - Parte 2 - Questao 3 - Traducao EN-DE com LangChain e HuggingFace")


@lru_cache(maxsize=None)
def get_chain():
    llm = HuggingFacePipeline.from_model_id(
        model_id=MODELO,
        task="translation",
        pipeline_kwargs={"max_length": 512},
    )
    return PromptTemplate.from_template("{text}") | llm | StrOutputParser()


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part for part in parts if part]


def translate_with_langchain_hf(text):
    chain = get_chain()
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return chain.invoke({"text": text}).strip()
    return " ".join(chain.invoke({"text": sentence}).strip() for sentence in sentences)


@app.get("/")
def root():
    return {"modelo": MODELO, "endpoint": "/translate", "metodo": "POST"}


@app.post("/translate")
def traduzir_com_huggingface(text: str = Body(..., embed=True, max_length=10000)):
    if not text.strip():
        raise HTTPException(status_code=400, detail="O campo text nao pode ser vazio.")
    try:
        traducao = translate_with_langchain_hf(text)
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Falha na traducao: {erro}")
    return {
        "modelo": MODELO,
        "orquestrador": "LangChain HuggingFacePipeline",
        "origem": "en",
        "destino": "de",
        "sentencas": len(split_sentences(text)),
        "texto_original": text,
        "texto_traduzido": traducao,
    }
