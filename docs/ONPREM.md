# On-Premises & Air-Gapped Deployment Guide

This guide details how to operate **JudgeLLM** in fully offline, sovereign, and air-gapped enterprise environments where **no data or prompts ever leave your physical or virtual perimeter**.

---

## 1. Architecture Overview

JudgeLLM provides a clean, provider-agnostic `JudgeBackend` abstraction layer:

```
┌────────────────────────────────────────────────────────┐
│                      JudgeLLM                          │
│  TeacherJudge • Eval Gate • Learned Router • Rubrics   │
└───────────────────────────┬────────────────────────────┘
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
   ┌───────────────────────┐  ┌───────────────────────┐
   │   vLLM Server         │  │   Ollama Server       │
   │   (OpenAI-Compatible) │  │   (OpenAI-Compatible) │
   │   http://localhost:8000│  │   http://localhost:11434│
   └───────────┬───────────┘  └───────────┬───────────┘
               │                          │
               └────────────┬─────────────┘
                            ▼
               ┌──────────────────────────┐
               │ Local GPU / Model Weight │
               │ Llama-3.3-70B / Qwen-72B │
               └──────────────────────────┘
```

The system uses standard HTTP `POST /v1/chat/completions` via Python standard library sockets (`urllib.request`), requiring **zero cloud dependencies** and **zero external telemetry**.

---

## 2. Prerequisites

- Python 3.11 installed locally
- Dedicated local accelerator (e.g. 1-4x NVIDIA A100/H100 80GB, or Apple Silicon Mac Studio for quantized testing)
- One of the following local inference runtimes:
  - **vLLM** (recommended for production high-throughput batch evaluation)
  - **Ollama** (recommended for workstation / rapid development)

---

## 3. Option A: Production Serving with vLLM

### 1. Download Model Weights Ahead of Time
In an internet-connected staging environment, mirror your target model weights (e.g. HuggingFace):
```bash
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct \
  --local-dir /opt/models/Llama-3.3-70B-Instruct
```
Transfer `/opt/models/` to your air-gapped machine.

### 2. Launch vLLM OpenAI-Compatible Server
Run vLLM bound strictly to localhost:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /opt/models/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 8192 \
  --enable-auto-tool-choice
```

---

## 4. Option B: Local Workstation Serving with Ollama

### 1. Pull Model
```bash
ollama pull llama3.3:70b
# or 8B for local testing:
ollama pull llama3.1:8b
```

### 2. Verify OpenAI-Compatible Endpoint
Ollama exposes `/v1` by default on `127.0.0.1:11434`:
```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.3:70b",
    "messages": [{"role": "user", "content": "Ping"}]
  }'
```

---

## 5. Configuring JudgeLLM for On-Premises Execution

You can configure JudgeLLM either via **YAML Configuration** or via **Environment Variables**.

### Via Configuration YAML

Use [`configs/eval_ollama_local.yaml`](file:///d:/AI/JudgeLLM/configs/eval_ollama_local.yaml) or [`configs/eval_vllm_onprem.yaml`](file:///d:/AI/JudgeLLM/configs/eval_vllm_onprem.yaml):

```yaml
judge_backend:
  type: openai_compatible
  base_url: http://localhost:8000/v1
  model_name: meta-llama/Llama-3.3-70B-Instruct
  api_key: local-airgap-token
  pricing:
    prompt_per_1k: 0.0
    completion_per_1k: 0.0

evaluation:
  rubric_path: evaluation/rubrics/rubric_b.yaml
  temperature: 0.1
  max_retries: 3
```

### Via Environment Variables

Set the following variables in `.env` or in your secure shell:
```bash
# Disable Groq / Cloud endpoints
GROQ_API_KEY=""

# Configure OpenAI-compatible local host
LOCAL_BASE_URL="http://127.0.0.1:8000/v1"
LOCAL_MODEL="meta-llama/Llama-3.3-70B-Instruct"
LOCAL_API_KEY="EMPTY"
JUDGE_BACKEND="openai_compatible"
```

---

## 6. Running Evaluations Completely Offline

### 1. Run Pipeline with On-Prem Config
```bash
.venv/bin/python run_pipeline.py --rubric evaluation/rubrics/rubric_b.yaml
```

### 2. Multi-Backend Comparison
Benchmark your local server against baseline datasets:
```bash
.venv/bin/python scripts/benchmark_backends.py --human data/human_annotations.json
```

### 3. Self-Preference Audit
Verify that your on-premise judge does not artificially favor local family outputs:
```bash
.venv/bin/python scripts/self_preference_test.py --backend openai_compatible --model llama3.3:70b
```

---

## 7. Security and Air-Gap Verification

To guarantee that zero packets leave your network:

1. **Firewall Rule Verification**:
   ```bash
   # Block all outbound TCP except to local subnet / loopback
   sudo iptables -A OUTPUT -d 127.0.0.1 -j ACCEPT
   sudo iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
   sudo iptables -P OUTPUT DROP
   ```

2. **Network Isolation Test**:
   ```bash
   # Confirm that JudgeLLM continues to evaluate 100% of samples without error
   .venv/bin/python -m pytest tests/test_judge_backends.py -v
   ```

All unit tests in `tests/test_judge_backends.py` use local mock/fake backends and require **zero network connectivity**.
