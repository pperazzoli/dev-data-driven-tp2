from fastapi import Body, FastAPI, HTTPException

from utils import generate_text, split_sentences, translate_en_fr

app = FastAPI(title="TP2 - Parte 1 - FastAPI + HuggingFace")


@app.get("/")
def root():
    return {
        "questao_1": {"endpoint": "/q1/generate", "modelo": "gpt2"},
        "questao_2": {"endpoint": "/q2/translate", "modelo": "Helsinki-NLP/opus-mt-en-fr"},
    }


@app.post("/q1/generate", tags=["Questao 1 - GPT-2"])
def gerar_texto(
    prompt: str = Body(..., embed=True, max_length=2000),
    max_new_tokens: int = Body(60, embed=True, ge=1, le=200),
):
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="O campo prompt nao pode ser vazio.")
    try:
        texto = generate_text(prompt, max_new_tokens=max_new_tokens)
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Falha na geracao de texto: {erro}")
    return {
        "modelo": "gpt2",
        "entrada": prompt,
        "texto_gerado": texto,
        "continuacao": texto[len(prompt):].strip(),
    }


@app.post("/q2/translate", tags=["Questao 2 - Traducao EN-FR"])
def traduzir_en_fr(text: str = Body(..., embed=True, max_length=10000)):
    if not text.strip():
        raise HTTPException(status_code=400, detail="O campo text nao pode ser vazio.")
    try:
        traducao = translate_en_fr(text)
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Falha na traducao: {erro}")
    return {
        "modelo": "Helsinki-NLP/opus-mt-en-fr",
        "origem": "en",
        "destino": "fr",
        "sentencas": len(split_sentences(text)),
        "texto_original": text,
        "texto_traduzido": traducao,
    }
