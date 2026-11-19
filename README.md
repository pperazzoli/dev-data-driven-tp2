# TP2 - FastAPI, HuggingFace e LangChain

## Instalacao

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt
```

O `requirements.txt` fixa `transformers>=4.40,<5` porque a versao 5 removeu o pipeline `translation`, usado nas questoes de traducao.

## Arquivos

| Arquivo | Conteudo |
|---|---|
| `utils.py` | funcoes compartilhadas pelas duas partes |
| `parte1.py` | Parte 1 - Q1 (GPT-2) e Q2 (traducao EN-FR com Helsinki-NLP) |
| `parte2.py` | Parte 2 - Q1 (Fake LLM), Q2 (LangChain + OpenAI) e Q3 (LangChain + HuggingFace EN-DE) |
| `respostas_parte1.pdf` | respostas teoricas da Parte 1 (Q3 e Q4) e diagrama da arquitetura |
| `respostas_parte2.pdf` | diagramas das Q1, Q2 e Q3, respostas das Q4 e Q5 e tabela comparativa da Q6 |
| `respostas_parte1.md` e `respostas_parte2.md` | texto-fonte dos PDFs |
| `diagramas.drawio` | diagramas e tabela comparativa editaveis no draw.io, uma pagina por questao |
| `figuras/` | os diagramas exportados em PNG, usados nos PDFs |

## Execucao

Cada parte e um app FastAPI. Rode um por vez e abra `http://127.0.0.1:8000/docs`.

```bash
uvicorn parte1:app --reload
```

```bash
uvicorn parte2:app --reload
```

| Parte | Endpoint | Questao |
|---|---|---|
| 1 | `POST /q1/generate` | geracao de texto com GPT-2 |
| 1 | `POST /q2/translate` | traducao EN-FR com Helsinki-NLP |
| 2 | `POST /q1/chat` | chatbot com Fake LLM |
| 2 | `GET /q1/perguntas` | perguntas que o Fake LLM conhece |
| 2 | `POST /q2/translate` | traducao EN-FR com LangChain + OpenAI |
| 2 | `POST /q3/translate` | traducao EN-DE com LangChain + HuggingFace |

A Q2 da Parte 2 precisa da chave da OpenAI na variavel de ambiente `OPENAI_API_KEY`. O modelo padrao e `gpt-4o-mini` e pode ser trocado pela variavel `OPENAI_MODEL`.

## Exemplos de requisicao

```bash
curl -X POST http://127.0.0.1:8000/q1/generate -H "Content-Type: application/json" -d "{\"prompt\": \"Artificial intelligence will\", \"max_new_tokens\": 50}"
```

```bash
curl -X POST http://127.0.0.1:8000/q2/translate -H "Content-Type: application/json" -d "{\"text\": \"Good morning. The weather is nice today.\"}"
```

```bash
curl -X POST http://127.0.0.1:8000/q1/chat -H "Content-Type: application/json" -d "{\"message\": \"O que e LangChain?\"}"
```
