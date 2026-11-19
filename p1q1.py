from functools import lru_cache

from fastapi import Body, FastAPI, HTTPException
from transformers import pipeline

MODELO = "gpt2"

app = FastAPI(title="TP2 - Parte 1 - Questao 1 - Geracao de texto com GPT-2")


@lru_cache(maxsize=None)
def get_generator():
    return pipeline("text-generation", model=MODELO)


def generate_text(prompt, max_new_tokens=60, temperature=0.8):
    generator = get_generator()
    result = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
        truncation=True,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    return result[0]["generated_text"]


@app.get("/")
def root():
    return {"modelo": MODELO, "endpoint": "/generate", "metodo": "POST"}


@app.post("/generate")
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
        "modelo": MODELO,
        "entrada": prompt,
        "texto_gerado": texto,
        "continuacao": texto[len(prompt):].strip(),
    }
