import re

from fastapi import Body, FastAPI, HTTPException
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

FAKE_ANSWERS = {
    "ola": "Ola! Eu sou um chatbot simulado com Fake LLM.",
    "oi": "Oi! Eu sou um chatbot simulado com Fake LLM.",
    "bom dia": "Bom dia! Como posso ajudar?",
    "quem e voce": "Sou um prototipo de chatbot construido com LangChain e um Fake LLM.",
    "qual seu nome": "Meu nome e FakeBot, um chatbot de testes.",
    "o que e langchain": "LangChain e um framework para orquestrar prompts, modelos e cadeias de processamento.",
    "o que e fastapi": "FastAPI e um framework web Python assincrono para construir APIs rapidas.",
    "o que e fake llm": "Fake LLM e um modelo falso que devolve respostas fixas, usado para testar a aplicacao sem custo.",
    "tchau": "Tchau! Ate a proxima.",
}

FALLBACK_ANSWER = "Ainda nao fui treinado para essa pergunta, sou apenas um Fake LLM de testes."

app = FastAPI(title="TP2 - Parte 2 - Questao 1 - Chatbot com Fake LLM")


def normalize(text):
    text = text.strip().lower()
    text = text.replace("á", "a").replace("ã", "a").replace("â", "a")
    text = text.replace("é", "e").replace("ê", "e")
    text = text.replace("í", "i").replace("ó", "o").replace("õ", "o")
    text = text.replace("ú", "u").replace("ç", "c")
    return re.sub(r"[^a-z0-9 ]", "", text)


def fake_answer_for(question):
    normalized = normalize(question)
    for key, answer in FAKE_ANSWERS.items():
        if key in normalized:
            return answer
    return FALLBACK_ANSWER


def fake_chat(question):
    llm = FakeListLLM(responses=[fake_answer_for(question)])
    prompt = PromptTemplate.from_template("Usuario: {question}\nAssistente:")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question}).strip()


@app.get("/")
def root():
    return {"llm": "FakeListLLM", "endpoint": "/chat", "metodo": "POST"}


@app.get("/perguntas")
def perguntas_conhecidas():
    return {"perguntas_conhecidas": list(FAKE_ANSWERS.keys())}


@app.post("/chat")
def chat(message: str = Body(..., embed=True, max_length=1000)):
    if not message.strip():
        raise HTTPException(status_code=400, detail="O campo message nao pode ser vazio.")
    return {"llm": "FakeListLLM", "pergunta": message, "resposta": fake_chat(message)}
