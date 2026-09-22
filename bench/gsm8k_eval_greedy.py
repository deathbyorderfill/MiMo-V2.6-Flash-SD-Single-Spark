#!/usr/bin/env python3
"""GSM8K eval against an OpenAI-compatible server.

Uses the gen params recorded in the checkpoint's shipped gsm8k_metrics.json
(temp 0.6, top_p 0.95, seed 0, max_tokens 8192). Prompt format: zero-shot chat
with '#### <answer>' instruction (standard for GSM8K chat evals).
"""
import concurrent.futures as cf
import json, os, re, sys, urllib.request

BASE = os.environ.get("BENCH_BASE", "http://127.0.0.1:8936/v1")
MODEL = os.environ.get("BENCH_MODEL", "mimo-sd")
N = int(os.environ.get("N", "300"))
THREADS = int(os.environ.get("THREADS", "8"))
OUT = os.environ.get("OUT", "gsm8k_greedy.json")

URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

PROMPT = ("Solve the following grade-school math problem. Reason step by step, "
          "then give the final numeric answer on its own line after '#### '.\n\n{q}")

ANS_RE = re.compile(r"####\s*\$?([\d,]+(?:\.\d+)?)")
NUM_RE = re.compile(r"-?\$?([\d,]+(?:\.\d+)?)")


def load():
    cache = "gsm8k_test.jsonl"
    if not os.path.exists(cache):
        with urllib.request.urlopen(URL, timeout=120) as r:
            open(cache, "wb").write(r.read())
    OFF = int(os.environ.get("OFFSET", "0")); rows = [json.loads(l) for l in open(cache)][OFF:OFF + N]
    return [(r["question"], r["answer"].split("####")[-1].strip().replace(",", "")) for r in rows]


def call(item):
    q, gold = item
    payload = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT.format(q=q)}],
               "max_tokens": 8192, "temperature": 0}
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            out = json.loads(r.read())
        text = out["choices"][0]["message"].get("content") or ""
        m = ANS_RE.search(text)
        if not m:
            nums = NUM_RE.findall(text)
            pred = nums[-1].replace(",", "") if nums else None
        else:
            pred = m.group(1).replace(",", "")
        ok = pred is not None and abs(float(pred) - float(gold)) < 1e-4
        return {"ok": ok, "pred": pred, "gold": gold, "error": None}
    except Exception as e:
        return {"ok": False, "pred": None, "gold": gold, "error": str(e)}


def main():
    items = load()
    results = []
    with cf.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for i, res in enumerate(ex.map(call, items), 1):
            results.append(res)
            if i % 25 == 0:
                done = [r for r in results]
                acc = sum(r["ok"] for r in done) / len(done)
                print(f"{i}/{len(items)} acc={acc:.4f}", flush=True)
    n_err = sum(1 for r in results if r["error"])
    acc = sum(r["ok"] for r in results) / len(results)
    summary = {"model": MODEL, "n": len(results), "errors": n_err, "score": acc,
               "baseline_nvfp4_score": 0.9727065959059894}
    with open(OUT, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
