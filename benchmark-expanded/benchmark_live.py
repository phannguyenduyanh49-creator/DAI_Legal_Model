"""Expanded LIVE API benchmark for DAI Legal Model. No synthetic results are generated."""
import csv, json, os, statistics, time, urllib.error, urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
URL=os.getenv("DAI_API_URL","http://localhost:8000/api/v1/chat/completions")
TOKEN=os.getenv("DAI_API_TOKEN","")
RUNS=max(1,int(os.getenv("DAI_BENCHMARK_RUNS","1")))
TIMEOUT=max(5,int(os.getenv("DAI_BENCHMARK_TIMEOUT","90")))
MODEL=os.getenv("DAI_MODEL_NAME","dai-legal-model")

def pct(values,p):
    if not values: return None
    v=sorted(values)
    return round(v[min(len(v)-1,int((p/100)*(len(v)-1)))],3)

def answer_text(obj):
    if not isinstance(obj,dict): return ""
    choices=obj.get("choices")
    if isinstance(choices,list) and choices:
        msg=choices[0].get("message",{})
        if isinstance(msg,dict): return msg.get("content","") or ""
        return choices[0].get("text","") or ""
    for k in ("answer","response","content"):
        if isinstance(obj.get(k),str): return obj[k]
    return ""

def request_api(question):
    payload={"model":MODEL,"messages":[{"role":"user","content":question}],"stream":False}
    headers={"Content-Type":"application/json"}
    if TOKEN: headers["Authorization"]="Bearer "+TOKEN
    req=urllib.request.Request(URL,data=json.dumps(payload,ensure_ascii=False).encode(),headers=headers,method="POST")
    start=time.perf_counter()
    try:
        with urllib.request.urlopen(req,timeout=TIMEOUT) as resp:
            raw=resp.read().decode("utf-8",errors="replace")
            elapsed=round((time.perf_counter()-start)*1000,3)
            try: obj=json.loads(raw)
            except json.JSONDecodeError: obj={}
            text=answer_text(obj)
            return {"http_status":resp.status,"latency_ms":elapsed,"success":200<=resp.status<300,
                    "answer_chars":len(text),"answer_preview":text[:250],"error":None}
    except urllib.error.HTTPError as e:
        return {"http_status":e.code,"latency_ms":round((time.perf_counter()-start)*1000,3),
                "success":False,"answer_chars":0,"answer_preview":"","error":f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"http_status":None,"latency_ms":round((time.perf_counter()-start)*1000,3),
                "success":False,"answer_chars":0,"answer_preview":"","error":f"{type(e).__name__}: {e}"}

def main():
    questions=json.loads((ROOT/"benchmark_questions.json").read_text(encoding="utf-8"))
    rows=[]; total=len(questions)*RUNS
    print(f"Endpoint={URL} | questions={len(questions)} | runs/question={RUNS} | requests={total}")
    print("Start with RUNS=1. This sends real API requests; check authorization and rate limits.")
    for i,q in enumerate(questions,1):
        for run in range(1,RUNS+1):
            result=request_api(q["question"])
            rows.append({"question_id":q["id"],"category":q["category"],"topic":q["topic"],
                         "run":run,"question":q["question"],**result})
            print(f"{len(rows)}/{total} {q['id']} status={result['http_status']} latency={result['latency_ms']}ms")
    ok=[r for r in rows if r["success"]]
    lat=[r["latency_ms"] for r in ok]
    errs=Counter(r["error"] for r in rows if not r["success"])
    cats=defaultdict(list)
    for r in ok: cats[r["category"]].append(r["latency_ms"])
    result={
      "project":"DAI Legal Model","generated_at_utc":datetime.now(timezone.utc).isoformat(),
      "classification":"LIVE_API_MEASUREMENTS","endpoint":URL,"question_count":len(questions),
      "runs_per_question":RUNS,"total_requests":len(rows),"success":len(ok),"failed":len(rows)-len(ok),
      "success_rate_percent":round(100*len(ok)/len(rows),2) if rows else 0,
      "latency_ms":{"mean":round(statistics.mean(lat),3) if lat else None,
                    "p50":pct(lat,50),"p95":pct(lat,95),
                    "min":min(lat) if lat else None,"max":max(lat) if lat else None},
      "category_mean_latency_ms":{k:round(statistics.mean(v),3) for k,v in sorted(cats.items())},
      "errors":dict(errs),"records":rows,
      "limitation":"Measures configured API end-to-end only; not isolated component timings or legal correctness."
    }
    (ROOT/"benchmark_live_results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    fields=list(rows[0].keys())
    with (ROOT/"benchmark_live_results.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    lines=["# DAI Legal Model - Expanded Live API Benchmark","",
      f"- Timestamp UTC: {result['generated_at_utc']}",f"- Endpoint: `{URL}`",
      f"- Questions: {len(questions)}",f"- Runs/question: {RUNS}",f"- Requests: {len(rows)}",
      f"- Success: {len(ok)}",f"- Failed: {len(rows)-len(ok)}",
      f"- Success rate: {result['success_rate_percent']}%","","## Latency (successful requests)",
      "| Metric | Value |","|---|---:|",
      f"| Mean | {result['latency_ms']['mean']} ms |",f"| P50 | {result['latency_ms']['p50']} ms |",
      f"| P95 | {result['latency_ms']['p95']} ms |",f"| Min | {result['latency_ms']['min']} ms |",
      f"| Max | {result['latency_ms']['max']} ms |","","## Category mean latency","",
      "| Category | Mean (ms) |","|---|---:|"]
    lines += [f"| {k} | {v} |" for k,v in result["category_mean_latency_ms"].items()]
    lines += ["","","## Errors",""]+[f"- {k}: {v}" for k,v in errs.items()]
    lines += ["","## Limitations","",
      "- Measures the configured Core API end-to-end; it does not isolate embedding, Qdrant, reranker, LLM or MCP timings.",
      "- Legal correctness and evidence grounding require a gold-standard evaluation set and human review.",
      "- Do not publish secrets or confidential user data.",""]
    (ROOT/"BENCHMARK_LIVE_REPORT.md").write_text("\\n".join(lines),encoding="utf-8")
    print(f"Completed: success={len(ok)} failed={len(rows)-len(ok)}")
    print("Generated benchmark_live_results.json, benchmark_live_results.csv, BENCHMARK_LIVE_REPORT.md")

if __name__=="__main__": main()
