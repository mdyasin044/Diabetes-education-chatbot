"""Build the vector store. Run once:  python ingest.py
Sources: public NIDDK/CDC type 2 diabetes pages + any PDF/TXT/MD you drop in ../knowledge.
Verify the URLs still resolve and that you may use each source before shipping."""
import glob, os
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from rag import embeddings, CHROMA_DIR

URLS = [
    "https://www.cdc.gov/diabetes/about/about-type-2-diabetes.html",
    "https://www.cdc.gov/diabetes/healthy-eating/index.html",
    "https://www.cdc.gov/diabetes/diabetes-complications/index.html",
    "https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes/type-2-diabetes",
    "https://www.niddk.nih.gov/health-information/diabetes/overview/managing-diabetes",
    "https://www.niddk.nih.gov/health-information/diabetes/overview/diet-eating-physical-activity",
    "https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-problems/low-blood-glucose-hypoglycemia",
    "https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-problems/high-blood-glucose-hyperglycemia",
]

docs = []
for url in URLS:
    try:
        d = WebBaseLoader(url).load()
        for x in d: x.metadata["source"] = url
        docs += d
        print("loaded", url)
    except Exception as e:
        print("skip", url, e)

for path in glob.glob("../knowledge/*"):
    if path.endswith(".pdf"): docs += PyPDFLoader(path).load()
    elif path.endswith((".txt", ".md")): docs += TextLoader(path, encoding="utf-8").load()

chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120).split_documents(docs)
chunks = [c for c in chunks if len(c.page_content.strip()) > 80]
Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
print(f"indexed {len(chunks)} chunks from {len(docs)} documents")
