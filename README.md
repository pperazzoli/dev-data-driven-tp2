# TP2 - DR3 - FastAPI, HuggingFace e LangChain

## Instalacao

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt
```

O `requirements.txt` fixa `transformers>=4.40,<5` porque a versao 5 removeu o pipeline `translation`, usado nas questoes de traducao.

## Arquivos

| Arquivo | Conteudo |
|---|---|
| `p1q1.py` | Parte 1, Questao 1 - geracao de texto com GPT-2 |
| `p1q2.py` | Parte 1, Questao 2 - traducao EN-FR com Helsinki-NLP |
| `p2q1.py` | Parte 2, Questao 1 - chatbot com Fake LLM |
| `p2q2.py` | Parte 2, Questao 2 - traducao EN-FR com LangChain + OpenAI |
| `p2q3.py` | Parte 2, Questao 3 - traducao EN-DE com LangChain + HuggingFace |
| `pedro_perazzoli_DR3_TP2.pdf` | link do repositorio, diagramas e analises solicitadas |
| `pedro_perazzoli_DR3_TP2.zip` | pacote de entrega: os cinco `.py` e o PDF |
| `respostas.md` | texto-fonte do PDF |
| `diagramas.drawio` | diagramas e tabela comparativa editaveis no draw.io, uma pagina por questao |
| `figuras/` | os diagramas exportados em PNG, usados no PDF |

## Execucao

Cada questao e um app FastAPI independente. Rode um por vez e abra `http://127.0.0.1:8000/docs`.

```bash
uvicorn p1q1:app --reload
```

```bash
uvicorn p1q2:app --reload
```

```bash
uvicorn p2q1:app --reload
```

```bash
uvicorn p2q2:app --reload
```

```bash
uvicorn p2q3:app --reload
```

| Arquivo | Endpoint | Corpo JSON |
|---|---|---|
| `p1q1.py` | `POST /generate` | `{"prompt": "...", "max_new_tokens": 60}` |
| `p1q2.py` | `POST /translate` | `{"text": "..."}` |
| `p2q1.py` | `POST /chat` e `GET /perguntas` | `{"message": "..."}` |
| `p2q2.py` | `POST /translate` | `{"text": "...", "target_language": "French"}` |
| `p2q3.py` | `POST /translate` | `{"text": "..."}` |

O `p2q2.py` precisa da chave da OpenAI na variavel de ambiente `OPENAI_API_KEY`. O modelo padrao e `gpt-4o-mini` e pode ser trocado pela variavel `OPENAI_MODEL`.

## Exemplos de requisicao

```bash
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d "{\"prompt\": \"Artificial intelligence will\", \"max_new_tokens\": 50}"
```

```bash
curl -X POST http://127.0.0.1:8000/translate -H "Content-Type: application/json" -d "{\"text\": \"Good morning. The weather is nice today.\"}"
```

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"O que e LangChain?\"}"
```
