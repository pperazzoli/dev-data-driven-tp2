from fastapi import Body, FastAPI, HTTPException

from utils import (
    FAKE_ANSWERS,
    OPENAI_MODEL,
    fake_chat,
    split_sentences,
    translate_with_chat_model,
    translate_with_langchain_hf,
)

MODELO_HF = "Helsinki-NLP/opus-mt-en-de"

app = FastAPI(title="TP2 - Parte 2 - LangChain")


@app.get("/")
def root():
    return {
        "questao_1": {"endpoint": "/q1/chat", "llm": "FakeListLLM"},
        "questao_2": {"endpoint": "/q2/translate", "modelo": OPENAI_MODEL},
        "questao_3": {"endpoint": "/q3/translate", "modelo": MODELO_HF},
    }


@app.get("/q1/perguntas", tags=["Questao 1 - Fake LLM"])
def perguntas_conhecidas():
    return {"perguntas_conhecidas": list(FAKE_ANSWERS.keys())}


@app.post("/q1/chat", tags=["Questao 1 - Fake LLM"])
def chat(message: str = Body(..., embed=True, max_length=1000)):
    if not message.strip():
        raise HTTPException(status_code=400, detail="O campo message nao pode ser vazio.")
    return {"llm": "FakeListLLM", "pergunta": message, "resposta": fake_chat(message)}


@app.post("/q2/translate", tags=["Questao 2 - OpenAI"])
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


@app.post("/q3/translate", tags=["Questao 3 - HuggingFace"])
def traduzir_com_huggingface(text: str = Body(..., embed=True, max_length=10000)):
    if not text.strip():
        raise HTTPException(status_code=400, detail="O campo text nao pode ser vazio.")
    try:
        traducao = translate_with_langchain_hf(text, model=MODELO_HF)
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Falha na traducao: {erro}")
    return {
        "modelo": MODELO_HF,
        "orquestrador": "LangChain HuggingFacePipeline",
        "origem": "en",
        "destino": "de",
        "sentencas": len(split_sentences(text)),
        "texto_original": text,
        "texto_traduzido": traducao,
    }
