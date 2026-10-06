"""Fetches user data. Mock mode uses local dummy data; real mode calls your REST API."""
import os, requests
import mock_data

USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"
BASE = os.getenv("DATA_API_BASE", "")

def _get(path):
    r = requests.get(f"{BASE}{path}", timeout=10)  # add auth headers here later
    r.raise_for_status()
    return r.json()

def get_foods():   return mock_data.mock_foods()   if USE_MOCK else _get("/foods?hours=12")
def get_glucose(): return mock_data.mock_glucose() if USE_MOCK else _get("/glucose?hours=12")
def get_watch():   return mock_data.mock_watch()   if USE_MOCK else _get("/watch?hours=12")
