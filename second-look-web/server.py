"""Serve one local checkpoint and a small, bounded inference API.

No checkpoint files, directory listings, or training data are served over HTTP.
"""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
import json
import hashlib
import logging
import os
from pathlib import Path
import re
import secrets
import time

from aiohttp import web

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("light_demo")
TTL = 600
MAX_PENDING = 4
OWNER = web.RequestKey("owner", str)


def build_prompt(config, values):
    # Same newline normalization as train_rewrite.make_build_prompt.
    return config["prompt_template"].format(**{
        key: value.strip().replace("\n", " ").replace("\r", " ")
        for key, value in values.items()
    })


class Inference:
    def __init__(self, config):
        self.config = config

    def load(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        self.torch = torch
        torch.set_num_threads(4)
        path = self.config["checkpoint"]
        device = self.config.get("device", "cuda:0")
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("Configured CUDA device is unavailable")
            free, _ = torch.cuda.mem_get_info(device)
            if free < 10 * 1024**3:
                raise RuntimeError("Less than 10 GiB free on the configured GPU")
        model_config = AutoConfig.from_pretrained(path, local_files_only=True)
        loader = AutoModelForImageTextToText if model_config.model_type == "mistral3" else AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = loader.from_pretrained(
            path, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": device},
        ).eval()
        LOG.info("Checkpoint loaded: %s", path)

    def generate(self, values):
        start = time.monotonic()
        prompt = build_prompt(self.config, values)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if inputs.input_ids.shape[1] > self.config.get("max_input_tokens", 512):
            raise ValueError("This text is too long for the demo. Please shorten it.")
        inputs = inputs.to(self.model.device)
        stops = self.config.get("stop_strings", [])
        stopping = {"stop_strings": stops, "tokenizer": self.tokenizer} if stops else {}
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.config.get("max_new_tokens", 768),
                max_time=75, pad_token_id=self.tokenizer.pad_token_id, use_cache=True, **stopping,
            )
        generated = output[0, inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        stopped_at_boundary = any(stop in text for stop in stops)
        for stop in stops:
            text = text.split(stop, 1)[0].strip()
        verdict = None
        if self.config.get("task") == "toxicity":
            # Read only the explicit final answer cue, never words in the explanation.
            answers = re.findall(r"My answer is:\s*(toxic|benign)\b", text, flags=re.I)
            verdict = answers[-1].lower() if answers else None
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        return {
            "output": text, "verdict": verdict,
            "seconds": round(time.monotonic() - start, 2),
            "tokens": len(generated),
            "truncated": bool(len(generated) and generated[-1].item() not in eos and not stopped_at_boundary),
        }


def create_app(config, engine=None):
    @web.middleware
    async def session(request, handler):
        token = request.cookies.get("second_look_session", "")
        fresh = re.fullmatch(r"[A-Za-z0-9_-]{43}", token) is None
        if fresh:
            token = secrets.token_urlsafe(32)
        request[OWNER] = hashlib.sha256(token.encode()).hexdigest()
        try:
            response = await handler(request)
        except web.HTTPException as exc:
            response = web.Response(status=exc.status, reason=exc.reason,
                                    headers=exc.headers, body=exc.body)
        if fresh:
            response.set_cookie("second_look_session", token, httponly=True, samesite="Strict",
                                secure=request.secure or request.headers.get("X-Forwarded-Proto") == "https",
                                max_age=86400, path="/")
        return response

    app = web.Application(client_max_size=16384, middlewares=[session])
    jobs = {}
    store = None
    if config.get("storage", {}).get("database"):
        from local_store import LocalStore
        store = LocalStore(ROOT / config["storage"]["database"])
    queue = asyncio.Queue(maxsize=MAX_PENDING)
    state = {"status": "loading"}
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
    training_status = config.get("startup_training_status") if engine is None else None
    if engine is None:
        if config.get("task") == "audited_assistant":
            from assistant_pipeline import AuditedAssistant
            engine = AuditedAssistant(config)
        else:
            engine = Inference(config)

    def prune():
        cutoff = time.time() - TTL
        for key in list(jobs):
            if jobs[key].get("finished", jobs[key]["created"]) < cutoff and jobs[key]["status"] in ("complete", "error"):
                del jobs[key]

    async def worker():
        loop = asyncio.get_running_loop()
        while training_status:
            try:
                training = json.loads(Path(training_status).read_text())
            except (OSError, ValueError):
                training = {}
            if training.get("state") == "complete":
                break
            if training.get("state") == "failed":
                state["status"] = "error"
                LOG.error("Alignment training did not complete; see its training log.")
                return
            await asyncio.sleep(5)
        try:
            await loop.run_in_executor(pool, engine.load)
        except Exception:
            LOG.exception("Model load failed")
            state["status"] = "error"
            return
        state["status"] = "ready"
        while True:
            key, values = await queue.get()
            job = jobs[key]
            job["status"] = "running"
            if store:
                store.put(key, job)
            try:
                if config.get("task") == "audited_assistant":
                    def save_progress(stage, current_job=job, current_key=key):
                        current_job["stage"] = stage
                        if store:
                            store.put(current_key, current_job)
                    def progress(stage):
                        loop.call_soon_threadsafe(save_progress, stage)
                    job["result"] = await loop.run_in_executor(pool, engine.generate, values, progress)
                else:
                    job["result"] = await loop.run_in_executor(pool, engine.generate, values)
                job["status"] = "complete"
            except ValueError as exc:
                job.update(status="error", error=str(exc))
            except Exception:
                LOG.exception("Inference failed")
                job.update(status="error", error="The model could not finish this request. Please try again.")
            finally:
                job["finished"] = time.time()
                if store:
                    store.put(key, job)
                values = None
                job = None
                queue.task_done()

    async def cleanup_results():
        while True:
            await asyncio.sleep(30)
            prune()

    async def lifecycle(_):
        task = asyncio.create_task(worker())
        cleanup = asyncio.create_task(cleanup_results())
        yield
        task.cancel()
        cleanup.cancel()
        with suppress(asyncio.CancelledError):
            await task
        with suppress(asyncio.CancelledError):
            await cleanup
        pool.shutdown(wait=False, cancel_futures=True)
        if store:
            store.close()

    async def headers(request, response):
        response.headers.update({
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        })

    async def info(_):
        return web.json_response({
            **{key: config[key] for key in ("title", "description", "model_label", "fields", "examples")},
            **state, "queued": queue.qsize(), "max_chars": 4000,
            "task": config.get("task", "toxicity"),
            "review_modes": config.get("review_modes", []),
            "default_review_mode": config.get("default_review_mode", "light"),
            "local_storage": store is not None,
            "public_path": config.get("public_path", "/"),
        })

    async def submit(request):
        if state["status"] != "ready":
            return web.json_response({"error": "The model is not ready yet."}, status=503)
        origin = request.headers.get("Origin")
        if origin:
            from urllib.parse import urlsplit
            if urlsplit(origin).netloc != request.host:
                return web.json_response({"error": "Use the demo page to submit text."}, status=403)
        if request.content_type != "application/json":
            return web.json_response({"error": "Expected JSON."}, status=415)
        try:
            data = await request.json()
        except (ValueError, UnicodeDecodeError):
            return web.json_response({"error": "Invalid JSON."}, status=400)
        keys = {field["name"] for field in config["fields"]}
        if isinstance(data, dict) and config.get("review_modes"):
            data = dict(data)
            data.setdefault("review_mode", config.get("default_review_mode", "light"))
            keys.add("review_mode")
            if not isinstance(data.get("review_mode"), str) or data["review_mode"] not in {mode["id"] for mode in config["review_modes"]}:
                return web.json_response({"error": "Choose Light or Fusion review."}, status=400)
        if not isinstance(data, dict) or set(data) != keys:
            return web.json_response({"error": "Please fill in the text fields."}, status=400)
        if any(not isinstance(value, str) or not value.strip() for value in data.values()):
            return web.json_response({"error": "Please enter some text."}, status=400)
        if sum(len(value) for key, value in data.items() if key != "review_mode") > 4000:
            return web.json_response({"error": "Please keep the input under 4,000 characters."}, status=400)
        prune()
        if queue.full() or sum(job["status"] in ("queued", "running") for job in jobs.values()) >= MAX_PENDING:
            return web.json_response({"error": "The demo is busy. Please try again shortly."}, status=429)
        pending = sum(job.get("owner") == request[OWNER] and job["status"] in ("queued", "running") for job in jobs.values())
        if pending >= config.get("max_pending_per_session", 1):
            return web.json_response({"error": "Please wait for an earlier response to finish before sending another message."}, status=429)
        key = secrets.token_urlsafe(24)
        jobs[key] = {"status": "queued", "created": time.time(), "owner": request[OWNER], "values": data}
        if store:
            store.put(key, jobs[key])
        queue.put_nowait((key, data))
        return web.json_response({"id": key, "status": "queued"}, status=202)

    async def result(request):
        prune()
        job = jobs.get(request.match_info["key"])
        if job is None and store:
            job = store.get(request.match_info["key"])
        if (job is None or job.get("owner") != request[OWNER] or
                (job["status"] in {"complete", "error"} and job.get("finished", job["created"]) < time.time() - TTL)):
            return web.json_response({"error": "This result has expired. Please run the text again."}, status=404)
        public = {key: value for key, value in job.items()
                  if key in {"status", "stage", "result", "error"}}
        if "result" in public and config.get("public_result_fields"):
            public["result"] = {k: v for k, v in public["result"].items()
                                if k in config["public_result_fields"]}
        return web.json_response(public)

    async def asset(request):
        # Explicit allowlist: never expose checkpoints, config, logs or arbitrary paths.
        name = request.match_info.get("name", "index.html")
        if name not in {"index.html", "app.js", "style.css"}:
            raise web.HTTPNotFound()
        return web.FileResponse(ROOT / config.get("static_directory", "static") / name)

    app.cleanup_ctx.append(lifecycle)
    app.on_response_prepare.append(headers)
    app.router.add_get("/api/info", info)
    app.router.add_post("/api/generate", submit)
    app.router.add_get("/api/results/{key}", result)
    public_path = config.get("public_path")
    if public_path and public_path != "/":
        app.router.add_get(public_path, asset)
        app.router.add_get(public_path + "/", asset)
    app.router.add_get("/", asset)
    app.router.add_get("/{name}", asset)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("gpu_index", 1))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    web.run_app(create_app(config), host="127.0.0.1", port=args.port, access_log=None)
