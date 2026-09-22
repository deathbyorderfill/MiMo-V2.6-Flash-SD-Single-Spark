"""Needle-in-a-haystack at a given token count against the OpenAI endpoint. Usage: needle.py <port> <model> <approx_tokens> [depth=0.5]"""
import json, random, sys, time, urllib.request
port, model, ntok = sys.argv[1], sys.argv[2], int(sys.argv[3]); depth = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
random.seed(ntok)
words = "the of and to in that is was for on with as by at from this be are or an it not which have has were their one all more".split()
n_words = int(ntok / 1.3)
body_words = [random.choice(words) for _ in range(n_words)]
code = f"{random.randint(100000, 999999)}"
needle = f" The secret access code for the vault is {code}. Remember it. "
pos = int(n_words * depth)
text = " ".join(body_words[:pos]) + needle + " ".join(body_words[pos:])
prompt = text + "\n\nQuestion: What is the secret access code for the vault? Answer with the number only."
req = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
t0 = time.time()
try:
    r = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}), timeout=5400).read())
    dt = time.time() - t0; ans = (r["choices"][0]["message"].get("content") or "").strip(); u = r.get("usage", {})
    print(f"NEEDLE tokens={u.get('prompt_tokens')} depth={depth} code={code} answer={ans[:60]!r} {'PASS' if code in ans else 'FAIL'} wall={dt:.0f}s prefill~{(u.get('prompt_tokens') or 0)/max(dt,1):.0f} tok/s", flush=True)
except Exception as e:
    body = getattr(e, "read", lambda: b"")(); print(f"NEEDLE tokens~{ntok} ERROR {type(e).__name__}: {str(e)[:120]} {body[:300]!r}", flush=True)
