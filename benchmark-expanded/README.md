# DAI Legal Model - Expanded Live Benchmark

- `benchmark_questions.json`: 100 Vietnamese questions across 10 categories.
- `benchmark_live.py`: sends real HTTP requests to the configured Core API.
- Output files are created only after the script is run: JSON, CSV, Markdown.

## PowerShell
Start with one run per question (100 real requests):

```powershell
$env:DAI_API_URL="http://localhost:8000/api/v1/chat/completions"
$env:DAI_BENCHMARK_RUNS="1"
python benchmark_live.py
```

After confirming the API request schema and service capacity, run 3 repetitions
per question (300 real requests):

```powershell
$env:DAI_BENCHMARK_RUNS="3"
python benchmark_live.py
```

Optional:
```powershell
$env:DAI_API_TOKEN="YOUR_TOKEN"
$env:DAI_BENCHMARK_TIMEOUT="90"
$env:DAI_MODEL_NAME="dai-legal-model"
```

The runner assumes an OpenAI-compatible chat-completions payload. Verify your
Core API accepts `model`, `messages`, and `stream`; adapt `request_api()` if
the project's actual API contract differs.

The script records actual HTTP status and elapsed time. It does not fabricate
successful responses or latency values. It measures API end-to-end only; separate
instrumentation is needed to isolate Qdrant, embedding, reranking, LLM and MCP.
Run only against an environment you are authorized to test.
