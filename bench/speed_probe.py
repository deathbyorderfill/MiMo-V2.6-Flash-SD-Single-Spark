#!/usr/bin/env python3
"""Decode-speed probe: novel prose / code / math prompts, greedy, streaming.
Reports decode tok/s (first token -> last token) per prompt single-stream, then a 3-stream aggregate.
Usage: speed_probe.py [tag]   (env PORT, default 8936; MAXTOK default 400)"""
import json, os, sys, time, threading, urllib.request
PORT = os.environ.get("PORT", "8936"); MAXTOK = int(os.environ.get("MAXTOK", "400"))
BASE = f"http://127.0.0.1:{PORT}/v1/chat/completions"
PROMPTS = {
    "prose": "Write a short story (about 300 words) about a lighthouse keeper on a volcanic island who discovers that the fog horn has started answering back.",
    "code": "Write a Python class implementing an LRU cache with O(1) get and put using a doubly linked list and a dict. Include type hints and a short usage example.",
    "math": "A tank holds 2400 litres. Pump A fills it at 18 litres per minute, pump B drains it at 7 litres per minute, and after 40 minutes pump C starts adding 5 litres per minute. Starting from empty, how long until the tank is full? Show each step.",
}


def gen(prompt, out, key):
    body = {"model": os.environ.get("MODEL", "mimo-sd"), "messages": [{"role": "user", "content": prompt}], "max_tokens": MAXTOK,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); tf = None; usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data:"): continue
            p = line[5:].strip()
            if p == b"[DONE]": continue
            d = json.loads(p)
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices") or []
            if ch and tf is None and ((ch[0].get("delta") or {}).get("content") or (ch[0].get("delta") or {}).get("reasoning_content")):
                tf = time.time()
    te = time.time(); ct = usage.get("completion_tokens", 0)
    out[key] = dict(ct=ct, ttft=(tf or te) - t0, dec=(ct - 1) / (te - tf) if tf and ct > 1 else 0.0, wall=te - t0)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    res = {}
    gen("Say hello.", res, "_warm")
    for k, p in PROMPTS.items():
        gen(p, res, k)
        print(f"[{tag}] single {k:5s}: {res[k]['ct']:4d} tok  ttft {res[k]['ttft']:.2f}s  decode {res[k]['dec']:.1f} tok/s", flush=True)
    multi = {}; ths = [threading.Thread(target=gen, args=(p, multi, k)) for k, p in PROMPTS.items()]
    t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]; wall = time.time() - t0
    tot = sum(v["ct"] for v in multi.values())
    print(f"[{tag}] 3 streams: {tot} tok in {wall:.1f}s = {tot / wall:.1f} tok/s aggregate; per stream " +
          " / ".join(f"{k} {v['dec']:.1f}" for k, v in multi.items()), flush=True)
    json.dump({"single": res, "multi": multi}, open(f"speed_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
