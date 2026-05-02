"""Quick test to verify Gemini and Groq API connectivity."""
import sys

from utils.config import get_config

# Force UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

results = []

cfg = get_config()

# Test 1: Gemini (NEW google-genai SDK)
results.append("[1] Testing Gemini API (google-genai SDK)...")
try:
    from google import genai

    key = cfg.gemini_api_key
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Reply with exactly one word: HELLO",
    )
    results.append(f"    RESULT: {response.text.strip()}")
    results.append("    STATUS: PASS")
except Exception as e:
    results.append(f"    ERROR: {type(e).__name__}: {str(e)[:200]}")
    results.append("    STATUS: FAIL")

# Test 2: Groq
results.append("[2] Testing Groq API...")
try:
    from groq import Groq

    key = cfg.groq_api_key
    client = Groq(api_key=key)
    response = client.chat.completions.create(
        model=cfg.models.fallback_llm,
        messages=[{"role": "user", "content": "Reply with exactly one word: HELLO"}],
        max_tokens=10,
    )
    results.append(f"    RESULT: {response.choices[0].message.content.strip()}")
    results.append("    STATUS: PASS")
except Exception as e:
    results.append(f"    ERROR: {type(e).__name__}: {str(e)[:200]}")
    results.append("    STATUS: FAIL")

# Test 3: Ollama
results.append("[3] Testing Ollama (local)...")
try:
    import ollama

    models = ollama.list()
    names = [m.get("name", m.get("model", "?")) for m in models.get("models", [])]
    results.append(f"    Models: {names}")
    results.append("    STATUS: PASS")
except Exception as e:
    results.append(f"    STATUS: SKIP ({type(e).__name__})")

# Write to file
with open("test_output.txt", "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n")
        print(line)
