# src/ollama_client.py
import requests, json

OLLAMA_URL = "http://localhost:11434"

def stream_chat(model: str, messages, options=None):
    """
    Generator yielding incremental text chunks from Ollama /api/chat.
    """
    payload = {"model": model, "messages": messages, "stream": True}
    if options:
        payload["options"] = options
    with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            if "message" in obj and "content" in obj["message"]:
                yield obj["message"]["content"]
            if obj.get("done"):
                break
