"""Run:  uvicorn main:app --reload --port 8000   (open http://localhost:8000)"""
from dotenv import load_dotenv
load_dotenv("../.env")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import context, rag, mock_data

app = FastAPI(title="Diabetes Education RAG")

class Msg(BaseModel):
    role: str
    content: str

class ChatIn(BaseModel):
    message: str
    history: list[Msg] = []

@app.get("/api/context")
def get_context():
    return context.build_context()

@app.post("/api/chat")
def chat(body: ChatIn):
    ctx = context.build_context()
    try:
        return rag.answer(body.message, [m.model_dump() for m in body.history],
                          context.to_prompt_text(ctx), ctx)
    except Exception as e:
        raise HTTPException(500, f"Chat failed: {e}")

# Mock versions of the REST APIs you'll build later (set USE_MOCK=false + DATA_API_BASE to switch)
@app.get("/mock/foods")   
def m_foods(hours: int = 12): return mock_data.mock_foods()
@app.get("/mock/glucose") 
def m_glucose(hours: int = 12): return mock_data.mock_glucose()
@app.get("/mock/watch")   
def m_watch(hours: int = 12): return mock_data.mock_watch()

app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
def index():
    return FileResponse("../frontend/index.html")
