# Deploy the Human-scale AI chatbot

The web UI, Python inference server, diagnosis prompts and process manager run on the same Linux host. The hosted deployment uses Python 3.10 and one NVIDIA GPU for all three BF16 models. The preflight checks for an Ampere-or-newer GPU and 18 GiB of free GPU memory. The hosted configuration selects physical GPU 1. Set `gpu_index` in `assistant_config.json` to the index on your host (usually 0 on a single-GPU machine). The selected device appears to the models as `cuda:0`.

## Install dependencies

Run these commands from the repository root on a host with an NVIDIA driver compatible with the CUDA 12.6 PyTorch wheels.

```bash
cd second-look-web
bash setup.sh
```

The installer creates `.venv` and installs the pinned dependencies. Run the remaining commands from `second-look-web/`.

## Checkpoints

Provide three complete Hugging Face checkpoint directories with weights, tokenizer and configuration files. The repository distributes the serving code. The host must supply the weights. You can use the paths below, symlink those directories, or edit the three `checkpoint` fields in [assistant_config.json](assistant_config.json).

| Directory | Checkpoint |
| --- | --- |
| `models/dialogue/` | [Llama 3.2 3B Instruct, deployed revision](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/tree/0cb88a4f764b7a12671c53f0838cd831a0843b95) |
| `models/light/` | Full-parameter Light reviewer trained from Llama 3.2 1B |
| `models/fusion/` | Full-parameter Fusion reviewer trained from the same base revision |

Both reviewers load at startup. To produce their weights from the training inputs included in this repository, see the [reviewer training recipe](../experiments/llama32_1b_chatbot_20261111/README.md). Model loading uses local files. Obtain access to the Llama checkpoints before starting.

## Start and verify

```bash
.venv/bin/python check_setup.py
.venv/bin/python -m unittest discover -v
.venv/bin/python manage.py start
.venv/bin/python manage.py status
curl -fsS http://127.0.0.1:8766/api/info
```

Loading continues in the background. When the API reports `"status": "ready"` and `"default_review_mode": "light"`, open the [local chat](http://127.0.0.1:8766/human-scale-chatbot-education) and send a short question. The page shows Thinking, Aligning and Responding, then displays the final answer. Select Fusion in the message box to use the other reviewer. Requests to `/api/generate` that omit `review_mode` use Light. An API client must retain its session cookie when polling for the result.

To run the browser checks against a running server, install the optional test dependency and browser.

```bash
.venv/bin/python -m pip install playwright==1.58.0
.venv/bin/python -m playwright install chromium
.venv/bin/python browser_controls_check.py --ui-only --url http://127.0.0.1:8766
```

This check simulates generation and verifies the default, manual selection and chat navigation. A regular submission in the chat exercises real inference.

## Public deployment

The hosted chatbot uses a Cloudflare Quick Tunnel that forwards to the loopback server on port 8766. To publish your own instance on Linux x86_64, run the following after the server is ready.

```bash
bash install_cloudflared.sh
.venv/bin/python manage.py tunnel
.venv/bin/python manage.py status
```

Open the reported `chat_url`. The assigned origin is also saved in `run/url.txt`. Restarting a Quick Tunnel can change the public hostname. To apply a backend or model configuration change while keeping the current tunnel running, use the following commands.

```bash
.venv/bin/python manage.py stop --server-only
.venv/bin/python manage.py start
.venv/bin/python manage.py status
```

Use `manage.py stop` to stop both processes. They survive a terminal disconnect and require startup after a host reboot. Startup and tunnel logs are in `run/server.log` and `run/tunnel.log`.

## Request storage

The public endpoint returns the final answer, elapsed time and truncation flag to the submitting browser session. The server saves prompts, drafts, diagnoses, model inputs and final replies in `data/second-look.sqlite3`. Browser result access expires after ten minutes. Local records persist until the operator removes them. Git ignores local records, logs, tunnel binaries and checkpoints.
