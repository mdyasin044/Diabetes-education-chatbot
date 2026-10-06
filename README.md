# Diabetes (T2D) education RAG chatbot

## Setup
cp .env.example .env        # add GEMINI_API_KEY
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ingest.py            # builds ./chroma_db from CDC/NIDDK pages (+ PDFs in ../knowledge)
uvicorn main:app --reload --port 8000
# open http://localhost:8000

## Swap in your real APIs
Set USE_MOCK=false and DATA_API_BASE in .env, then match the shapes in backend/mock_data.py
(GET /foods, /glucose, /watch with ?hours=12), or edit backend/data_client.py.
