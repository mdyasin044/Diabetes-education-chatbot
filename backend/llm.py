"""LangChain wrapper around the Gemini API generateContent endpoint (Gemma model)."""
import os, re, requests
from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM

API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("GEMINI_MODEL", "gemma-4-26b-a4b-it")
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

def strip_markdown(s: str) -> str:
    s = re.sub(r"[*_#`]", "", s)
    s = re.sub(r"^\s*[-•]\s+", "", s, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", s).strip()

class GemmaLLM(LLM):
    system_instruction: str = ""
    temperature: float = 0.3
    max_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        return "gemma-generate-content"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager=None, **kwargs: Any) -> str:
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature, "maxOutputTokens": self.max_tokens,
                                 "thinkingConfig": {"thinkingLevel": "MINIMAL"}},
        }
        if self.system_instruction:
            body["system_instruction"] = {"parts": [{"text": self.system_instruction}]}
        r = requests.post(ENDPOINT, json=body, timeout=90)
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts if not p.get("thought") and "text" in p)
        return strip_markdown(text)