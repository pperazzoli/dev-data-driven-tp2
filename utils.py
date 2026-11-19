import os
import re
from functools import lru_cache

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


def normalize(text):
    text = text.strip().lower()
    text = text.replace("á", "a").replace("ã", "a").replace("â", "a")
    text = text.replace("é", "e").replace("ê", "e")
    text = text.replace("í", "i").replace("ó", "o").replace("õ", "o")
    text = text.replace("ú", "u").replace("ç", "c")
    return re.sub(r"[^a-z0-9 ]", "", text)


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part for part in parts if part]


@lru_cache(maxsize=None)
def get_pipeline(task, model):
    from transformers import pipeline
    return pipeline(task, model=model)


def generate_text(prompt, max_new_tokens=60, temperature=0.8):
    generator = get_pipeline("text-generation", "gpt2")
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


def translate(text, model):
    translator = get_pipeline("translation", model)
    return translator(text, max_length=512)[0]["translation_text"]


def translate_long(text, model):
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return translate(text, model)
    return " ".join(translate(sentence, model) for sentence in sentences)


def translate_en_fr(text):
    return translate_long(text, "Helsinki-NLP/opus-mt-en-fr")


def fake_answer_for(question):
    normalized = normalize(question)
    for key, answer in FAKE_ANSWERS.items():
        if key in normalized:
            return answer
    return FALLBACK_ANSWER


def fake_chat(question):
    from langchain_core.language_models.fake import FakeListLLM
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    llm = FakeListLLM(responses=[fake_answer_for(question)])
    prompt = PromptTemplate.from_template("Usuario: {question}\nAssistente:")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question}).strip()


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@lru_cache(maxsize=None)
def get_chat_model():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0)


def translate_with_chat_model(text, target_language="French"):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional translator. Translate the user text into {target_language}. Answer only with the translation, keeping the original formatting."),
        ("human", "{text}"),
    ])
    chain = prompt | get_chat_model() | StrOutputParser()
    return chain.invoke({"text": text, "target_language": target_language}).strip()


@lru_cache(maxsize=None)
def get_hf_langchain_llm(model):
    from langchain_huggingface import HuggingFacePipeline
    return HuggingFacePipeline.from_model_id(
        model_id=model,
        task="translation",
        pipeline_kwargs={"max_length": 512},
    )


def translate_with_langchain_hf(text, model="Helsinki-NLP/opus-mt-en-de"):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    chain = PromptTemplate.from_template("{text}") | get_hf_langchain_llm(model) | StrOutputParser()
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return chain.invoke({"text": text}).strip()
    return " ".join(chain.invoke({"text": sentence}).strip() for sentence in sentences)
