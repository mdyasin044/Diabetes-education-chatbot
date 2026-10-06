"""LangChain RAG chain: personal 12h context + retrieved T2D education passages."""
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm import GemmaLLM

CHROMA_DIR = "./chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

SYSTEM = (
    "You are a diabetes education assistant for an adult with type 2 diabetes. "
    "Explain glucose trends and answer questions about nutrition, activity, sleep and medication in general terms. "
    "Use the USER DATA to link meals, steps, heart rate and glucose. Use the SOURCES for medical facts and cite them like [1]. "
    "If the sources don't cover something, say so. "
    "Never give insulin or medication doses, never tell the user to start, stop or change a medicine, and never diagnose. "
    "For treatment decisions, tell them to follow their care plan or contact their care team. "
    "If the user asks anything unrelated to diabetes, politely decline. "
    "Output ONLY the direct reply to the user, with no reasoning or notes. "
    "Do NOT begin with phrases like 'The user said'. "
    "Do NOT use markdown, asterisks, bullet points, headers, or bold/italic. "
    "Keep replies under 80 words, in plain conversational sentences."
)

PROMPT = ChatPromptTemplate.from_template("""USER DATA (last 12 hours):
{user_data}

SOURCES:
{sources}

CONVERSATION SO FAR:
{history}

User question: {question}""")

_chain = PROMPT | GemmaLLM(system_instruction=SYSTEM) | StrOutputParser()
_store = None

def retriever():
    global _store
    if _store is None:
        _store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return _store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 15})

def safety_alerts(ctx, question):
    """Hard-coded rules that don't depend on the LLM."""
    alerts, g = [], ctx["glucose"]["latest"]
    if g < 70:
        alerts.append(f"Your latest glucose is {g} mg/dL, which is low. Follow your hypoglycemia plan now. If you feel confused, faint, or can't swallow, call emergency services.")
    elif g > 300:
        alerts.append(f"Your latest glucose is {g} mg/dL, which is very high. Contact your care team. If you have vomiting, trouble breathing, or confusion, seek urgent care.")
    if re.search(r"\b(chest pain|can'?t breathe|unconscious|passed out|vomiting|confus)", question, re.I):
        alerts.append("Those symptoms can be serious. Please contact your care team or emergency services now.")
    return alerts

def refuses_dosing(question):
    return bool(re.search(r"\b(how (much|many)|what|should i).{0,40}(insulin|units|dose|dosage|metformin|medication)", question, re.I))

def answer(question, history, user_text, ctx):
    alerts = safety_alerts(ctx, question)
    if refuses_dosing(question):
        return {"answer": "I can't advise on doses or medication changes. Please ask your care team. I can explain how food, activity, and sleep affect your glucose if that helps.",
                "sources": [], "alerts": alerts}
    docs = retriever().invoke(f"{question} type 2 diabetes {ctx['glucose']['trend_15min']} glucose")
    sources = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
    hist = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:]) or "(none)"
    text = _chain.invoke({"user_data": user_text, "sources": sources, "history": hist, "question": question})
    seen, links = set(), []
    for i, d in enumerate(docs):
        u = d.metadata.get("source", "")
        if u and u not in seen:
            seen.add(u); links.append({"n": i + 1, "url": u})
    return {"answer": text, "sources": links, "alerts": alerts}
