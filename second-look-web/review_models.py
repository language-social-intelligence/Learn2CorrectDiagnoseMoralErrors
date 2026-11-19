"""Full Llama alignment checkpoints, with the legacy adapter loader retained.

The HTTP worker serializes requests, including adapter selection and generation.
Fusion diagnosis includes steps 1--5; step 6 is correction, not diagnosis.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def parse_review(result, mode):
    if mode not in {"light", "fusion"}:
        raise ValueError("Choose Light or Fusion review.")
    raw = result.get("output", "").strip()
    number = "2" if mode == "light" else "6"
    marker = re.search(
        rf"(?:^|\s)(?:\({number}\)|{number}\.\s+)"
        r"|#{2,}\s*(?:Correction|Revision|Revised\s*Reply)\b"
        r"|<\s*(?:Revised|Corrected|Suggested|Final|(?:Morally\s*)?(?:Acceptable|Appropriate))\s*(?:Reply|Response)\s*>"
        r"|(?:^|\s)(?:Therefore,?\s*)?(?:the\s+)?Revised\s*Reply\s*(?:is\b|:)",
        raw, re.I,
    )
    if not raw or (result.get("truncated") and marker is None):
        raise ValueError("The reviewer did not finish its diagnosis. Please try a shorter question.")
    diagnosis = raw[:marker.start()] if marker else raw
    diagnosis = re.sub(r"^\s*(?:###\s*Diagnosis:\s*)?(?:\(1\)|1\.)?\s*", "", diagnosis, flags=re.I).strip()
    if not diagnosis or diagnosis.lower().strip('." ') in {"toxic", "benign"}:
        raise ValueError("The reviewer returned no diagnosis. Please try again.")
    return {"diagnosis": diagnosis, "raw_output": raw,
            "tokens": result.get("tokens"), "review_mode": mode}


class FullReviewModels:
    """Separate, fully fine-tuned Llama checkpoints; no adapters at inference."""

    def __init__(self, config):
        self.config = config

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(4)
        self.models = {}
        self.prompts = {}
        for mode, cfg in self.config['adapters'].items():
            path = Path(cfg['checkpoint'])
            if (path/'adapter_config.json').exists():
                raise RuntimeError('This deployment requires full model weights, not an unmerged adapter.')
            model, info = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True,
                dtype=torch.bfloat16, device_map={'': self.config['device']},
                attn_implementation='sdpa', output_loading_info=True)
            if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
                raise RuntimeError(f'{mode}: the full checkpoint did not load cleanly.')
            if model.config.model_type != 'llama' or model.config.hidden_size != 2048:
                raise RuntimeError(f'{mode}: expected a Llama 3.2 1B checkpoint.')
            self.models[mode] = model.requires_grad_(False).eval()
            # Training records are single lines; preserve their whitespace convention.
            self.prompts[mode] = (ROOT/cfg['prompt_file']).read_text().replace('\n', ' ').rstrip()
        first = next(iter(self.config['adapters'].values()))
        self.tokenizer = AutoTokenizer.from_pretrained(first['checkpoint'], local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def diagnose(self, question, answer, mode='fusion'):
        if mode not in self.models:
            raise ValueError('The requested alignment model is unavailable.')
        cfg = self.config['adapters'][mode]
        model = self.models[mode]
        prompt = self.prompts[mode].format(**{
            k:v.strip().replace('\n',' ').replace('\r',' ')
            for k,v in {'question':question,'answer':answer}.items()})
        inputs = self.tokenizer(prompt, return_tensors='pt')
        if inputs['input_ids'].shape[1] > self.config.get('max_input_tokens',4096):
            raise ValueError('The message is too long for alignment. Please shorten it.')
        inputs = inputs.to(model.device)
        stops = ['(2)' if mode == 'light' else '(6)', '<RevisedReply>', '<Revised Reply>',
                 '<MorallyAcceptableReply>', '<MorallyAppropriateReply>', '<CorrectedReply>',
                 '<SuggestedReply>', '###Revision', '###Correction', '<Prompt>:']
        with self.torch.inference_mode():
            output = model.generate(**inputs, do_sample=False,
                max_new_tokens=cfg.get('max_new_tokens',512), max_time=90,
                pad_token_id=self.tokenizer.pad_token_id, stop_strings=stops,
                tokenizer=self.tokenizer, use_cache=True)
        generated = output[0,inputs['input_ids'].shape[1]:]
        raw = self.tokenizer.decode(generated,skip_special_tokens=True).strip()
        eos = model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        eos = eos if isinstance(eos,list) else [eos]
        boundary = any(s in raw for s in stops)
        try:
            parsed = parse_review({'output':raw,'tokens':len(generated),
                'truncated':bool(len(generated) and generated[-1].item() not in eos and not boundary)},mode)
        except ValueError as exc:
            raise ValueError("I couldn't finish the response. Please try a shorter question.") from exc
        parsed['active_adapter'] = mode
        parsed['input_text'] = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
        return parsed


class LocalReviewModels:
    def __init__(self, config):
        self.config = config

    def load(self):
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer, Mistral3ForConditionalGeneration
        self.torch = torch
        path = self.config["base_checkpoint"]
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, fix_mistral_regex=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = Mistral3ForConditionalGeneration.from_pretrained(
            path, local_files_only=True, dtype=torch.bfloat16,
            device_map={"": self.config["device"]}, attn_implementation="sdpa",
        )
        adapters = self.config["adapters"]
        self.model = PeftModel.from_pretrained(
            base, adapters["light"]["checkpoint"], adapter_name="light", is_trainable=False,
            local_files_only=True,
        )
        self.model.load_adapter(adapters["fusion"]["checkpoint"], adapter_name="fusion",
                                is_trainable=False, local_files_only=True)
        self.model.requires_grad_(False).eval()
        # Validate actual tensor contents, not just adapter names, across runtime versions.
        from safetensors import safe_open
        for mode, cfg in adapters.items():
            with safe_open(str(Path(cfg["checkpoint"])/"adapter_model.safetensors"), framework="pt") as saved:
                actual = {name.replace(f".{mode}.", "."): value
                          for name, value in self.model.named_parameters()
                          if "lora_" in name and f".{mode}." in name}
                if set(saved.keys()) != set(actual):
                    raise RuntimeError(f"{mode}: loaded adapter parameter names do not match the trained checkpoint.")
                for name in saved.keys():
                    if not torch.equal(actual[name].detach().cpu(), saved.get_tensor(name)):
                        raise RuntimeError(f"{mode}: loaded adapter weights differ from the trained checkpoint.")
        self.prompts = {mode: (ROOT / cfg["prompt_file"]).read_text()
                        for mode, cfg in adapters.items()}

    def diagnose(self, question, answer, mode="light"):
        if mode not in self.config["adapters"]:
            raise ValueError("Choose Light or Fusion review.")
        cfg = self.config["adapters"][mode]
        values = {"question": question, "answer": answer}
        prompt = self.prompts[mode].format(**{
            k: v.strip().replace("\n", " ").replace("\r", " ") for k, v in values.items()
        })
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if inputs["input_ids"].shape[1] > self.config.get("max_input_tokens", 4096):
            raise ValueError("The question and draft are too long for review. Please shorten the question.")
        inputs = inputs.to(self.model.device)
        self.model.set_adapter(mode)
        if self.model.active_adapters != [mode]:
            raise RuntimeError("The requested review adapter was not activated.")
        self.model.requires_grad_(False).eval()
        stop = "(2)" if mode == "light" else "(6)"
        stops = [stop, "<RevisedReply>", "<Revised Reply>", "<MorallyAcceptableReply>",
                 "<MorallyAppropriateReply>", "<CorrectedReply>", "<SuggestedReply>",
                 "###Revision", "###Correction", "<Prompt>:"]
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=cfg["max_new_tokens"],
                max_time=120, pad_token_id=self.tokenizer.pad_token_id,
                stop_strings=stops, tokenizer=self.tokenizer, use_cache=True,
            )
        generated = output[0, inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        eos = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        boundary = any(stop in raw for stop in stops)
        raw = raw.split("<Prompt>:", 1)[0].strip()
        result = parse_review({"output": raw, "tokens": len(generated),
                               "truncated": bool(len(generated) and generated[-1].item() not in eos and not boundary)}, mode)
        result["active_adapter"] = self.model.active_adapter
        return result
