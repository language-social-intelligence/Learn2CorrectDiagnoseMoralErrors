"""Local answer -> selected light/fusion diagnosis -> one local rewrite.

See docs/METHOD.md for source mapping and application-specific extensions.
No model download or substitute model is selected automatically.
"""
from pathlib import Path
import re
import time

from server import Inference


SYSTEM_PROMPT = """You are a helpful general-purpose AI assistant. Answer the user's question clearly and accurately. State uncertainty and do not invent facts, citations or capabilities. Do not provide instructions that facilitate harm.
For health questions, provide general health information, not a definitive medical diagnosis or individualized treatment plan. You are not a clinician. Encourage a qualified healthcare professional for diagnosis and treatment decisions. If the user describes an apparent medical emergency, prioritize seeking urgent local medical help. Never imply that an automated language audit verifies medical accuracy.
Keep the answer focused and concise, normally within 150 words."""

def correction_prompt(question, answer, diagnosis):
    # Diagnosis-guided external correction from ../prompt.md, semantic Turn-2
    # (also used for Fusion in ../myMain_zi.py), with the requested line breaks.
    question = question.strip().replace("\n", " ")
    answer = answer.strip().replace("\n", " ")
    return (
        f'<Prompt>: "{question}"; <Reply>: "{answer}". Here is a diagnosis.'
        f'\n###Diagnosis: {diagnosis}\n'
        '\nBased on the ###Diagnosis above, please rewrite the Reply.\n'
        'Only give the revised Reply without any explanation.\n'
        'My Revised Reply is: '
    )


def base_dialogue_prompt(question):
    """Completion prompting for the explicitly requested pretrained base model."""
    chinese = bool(re.search(r'[\u4e00-\u9fff]', question))
    examples = (
        'User: 什么是光合作用？\n'
        'Assistant: 光合作用是植物利用阳光将水和二氧化碳转化为糖并释放氧气的过程。\n\n'
        'User: 怎样礼貌地表达不同意见？\n'
        'Assistant: 可以说：“我理解你的想法。我有另一个角度，我们可以一起比较一下吗？”\n\n'
        'User: 如何感谢同事的帮助？\n'
        'Assistant: 谢谢你的帮助，你的支持让我顺利完成了这项工作。\n\n'
        if chinese else
        'User: What is photosynthesis?\n'
        'Assistant: Photosynthesis is how plants use sunlight to turn water and carbon dioxide into sugars, releasing oxygen.\n\n'
        'User: How can I politely disagree with a colleague?\n'
        'Assistant: You could say, "I see your point. I have a different perspective—could we compare our approaches?"\n\n'
        'User: Write one sentence thanking someone for their support.\n'
        'Assistant: Thank you for your help; your support made a real difference.\n\n'
    )
    return (
        'A conversation with Human-scale AI, a helpful assistant. '
        'The assistant gives clear, accurate, concise answers in the user\'s language and acknowledges uncertainty.\n\n'
        f'{examples}'
        f'User: {question.strip()}\nAssistant: '
    )



def parse_light_diagnosis(result):
    raw = result.get("output", "").strip()
    if not raw or result.get("truncated"):
        raise ValueError("The light model did not finish its diagnosis. Please try again.")
    # Retain only step (1), discarding step (2) and any revised-reply text.
    marker = re.search(
        r'(?:^|\s)(?:###\s*(?:Correction|Revision|Revised Reply)\s*:\s*|'
        r'\(2\)\s*|2\.\s+|(?:Therefore,?\s*)?(?:the\s+)?<Revised Reply>\s+is\s*:?\s*|'
        r'The utterance is toxic or benign\?\s*My answer is:)',
        raw, re.I,
    )
    diagnosis = re.sub(r'^\s*(?:###\s*Diagnosis:\s*)?(?:\(1\)|1\.)?\s*', '', raw[:marker.start()] if marker else raw, flags=re.I).strip()
    if not diagnosis or diagnosis.lower().strip('." ') in {"toxic", "benign"}:
        raise ValueError("The light model returned no diagnosis.")
    return {"diagnosis": diagnosis, "raw_output": raw}


def parse_muse_answer(raw):
    """Extract only Muse's user-directed answer, never its self/tool channel.

    The built-in template ends at <|start|>assistant, so the first generated
    segment continues that header. Preserve special tokens until after parsing.
    """
    transcript = "<|start|>assistant" + raw
    answers = []
    for match in re.finditer(
        r"<\|start\|>assistant(?:\s+to=([^\s<]+))?\s*<\|message\|>"
        r"(.*?)(?=<\|start\|>|$)", transcript, re.S,
    ):
        if match.group(1) not in (None, "user"):
            continue
        body = re.split(r"<\|(?:eot|eom|end_of_text)\|>", match.group(2), maxsplit=1)[0].strip()
        if body and "<atem:" not in body:
            answers.append(body)
    if not answers:
        raise ValueError("Muse did not reach a final answer within the generation limit. Please try a shorter question.")
    return "\n\n".join(answers)


class LocalLightDiagnoser:
    def __init__(self, config):
        self.config = config
        engine_config = dict(config)
        if config.get("prompt_file"):
            engine_config["prompt_template"] = (Path(__file__).parent / config["prompt_file"]).read_text().rstrip() + " "
        self.engine = Inference(engine_config)

    def load(self):
        if not self.config.get("checkpoint") or self.config.get("output_contract") != "diagnosis":
            raise RuntimeError("Configure a light checkpoint and its diagnosis prompt.")
        self.engine.load()

    def diagnose(self, question, answer):
        return parse_light_diagnosis(self.engine.generate({"question": question, "answer": answer}))


class LocalChatModel:
    def __init__(self, config):
        self.config = config

    def load(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        self.torch = torch
        torch.set_num_threads(4)
        path = self.config.get("checkpoint")
        if not path or not Path(path).is_dir():
            raise RuntimeError("Main model is not configured: choose its exact model ID and local checkpoint first.")
        devices = self.config.get("devices") or [self.config["device"]]
        for device in devices:
            free, _ = torch.cuda.mem_get_info(device)
            if free < self.config.get("minimum_free_gib", 65) * 1024**3:
                raise RuntimeError(f"{device} does not have enough free memory for the main model.")
        model_config = AutoConfig.from_pretrained(path, local_files_only=True)
        self.is_muse = model_config.model_type == "muse_glimmer"
        if (Path(path) / "tekken.json").exists() and not (Path(path) / "tokenizer.json").exists():
            from transformers import MistralCommonBackend
            self.tokenizer = MistralCommonBackend.from_pretrained(path, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True,
                **({"fix_mistral_regex": True} if model_config.model_type == "mistral3" else {}))
        if self.is_muse:
            from transformers import AutoModelForMultimodalLM
            loader = AutoModelForMultimodalLM
        else:
            loader = AutoModelForImageTextToText if model_config.model_type == "mistral3" else AutoModelForCausalLM
        placement = {"device_map": {"": devices[0]}}
        if len(devices) > 1:
            placement = {
                "device_map": "balanced",
                "max_memory": {int(d.split(":")[1]): f'{self.config["memory_per_gpu_gib"]}GiB' for d in devices},
            }
        if self.config.get("quantization") == "int8":
            from transformers import BitsAndBytesConfig
            placement["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.config.get("quantization") == "nf4":
            from transformers import BitsAndBytesConfig
            placement["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.model, loading_info = loader.from_pretrained(
            path, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", output_loading_info=True, **placement,
        )
        if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError("Main checkpoint did not load cleanly; refusing to serve partially initialized weights.")
        self.model.eval()

    def prepare_inputs(self, question, *, system=SYSTEM_PROMPT):
        messages = [] if system is None else [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": question})
        base_mode = self.config.get("prompt_mode") == "base_dialogue"
        if base_mode:
            # The correction call already contains the complete research prompt.
            prompt = question if system is None else base_dialogue_prompt(question)
            inputs = self.tokenizer(prompt, return_tensors="pt")
        elif self.config.get("prompt_mode") == "llama_instruct":
            assistant_prefix = 'My Revised Reply is: '
            suffix = '\n' + assistant_prefix
            if system is None and question.endswith(suffix):
                # Match ../myMain_zi.py's Llama off-the-shelf correction chat.
                user_content = question[:-len(suffix)]
                prompt = (
                    f'{self.tokenizer.bos_token}'
                    '<|start_header_id|>user<|end_header_id|>\n\n'
                    f'{user_content}<|eot_id|>\n'
                    '<|start_header_id|>assistant<|end_header_id|>\n\n'
                    f'{assistant_prefix}'
                )
            else:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            # The rendered chat already contains its BOS and role tokens.
            inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        else:
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
                **({"reasoning_strength": self.config.get("reasoning_strength", "low")} if self.is_muse else {}),
            )
        return inputs

    def chat(self, question, *, system=SYSTEM_PROMPT):
        inputs = self.prepare_inputs(question, system=system)
        input_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
        base_mode = self.config.get("prompt_mode") == "base_dialogue"
        if inputs["input_ids"].shape[1] > self.config.get("max_input_tokens", 6144):
            raise ValueError("The question and audit exceed this demo's context limit. Please shorten the question.")
        inputs = inputs.to(self.model.get_input_embeddings().weight.device)
        sampling = {"do_sample": False}
        if self.config.get("temperature", 0) > 0:
            sampling = {"do_sample": True, "temperature": self.config["temperature"],
                        "top_p": self.config.get("top_p", 0.95), "top_k": self.config.get("top_k", 64)}
        with self.torch.inference_mode():
            stops = ["\nUser:", "\nHuman:", "\nAssistant:", "<Prompt>:"] if base_mode else []
            output = self.model.generate(
                **inputs, **sampling,
                max_new_tokens=self.config.get("max_new_tokens", 512),
                max_time=self.config.get("max_time", 90),
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                use_cache=True,
                **({"stop_strings": stops, "tokenizer": self.tokenizer} if stops else {}),
            )
        generated = output[0, inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=not self.is_muse).strip()
        text = parse_muse_answer(raw) if self.is_muse else raw
        boundary = any(stop in text for stop in stops)
        for stop in stops:
            text = text.split(stop, 1)[0].strip()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        return {"text": text, "raw_output": raw, "input_text": input_text,
                "input_tokens": inputs['input_ids'].shape[1], "tokens": len(generated),
                "truncated": bool(len(generated) and generated[-1].item() not in eos and not boundary)}


class AuditedAssistant:
    def __init__(self, config, main=None, light=None, reviewers=None):
        self.config = config
        self.main = main or LocalChatModel(config["main_model"])
        self.reviewers = reviewers
        if "review_model" in config and light is None and reviewers is None:
            from review_models import LocalReviewModels, FullReviewModels
            cls = FullReviewModels if config["review_model"].get("format") == "full" else LocalReviewModels
            self.reviewers = cls(config["review_model"])
        self.light = light or (None if self.reviewers else LocalLightDiagnoser(config["light_model"]))

    def load(self):
        self.main.load()
        (self.reviewers or self.light).load()

    def generate(self, values, progress=lambda stage: None):
        started = time.monotonic()
        question = values["question"].strip()
        mode = values.get("review_mode", self.config.get("default_review_mode", "light"))
        if mode not in {"light", "fusion"}:
            raise ValueError("Choose Light or Fusion review.")
        progress("generating")
        initial = self.main.chat(question)
        if not initial["text"]:
            raise ValueError("The assistant returned an empty answer. Please try again.")
        progress("diagnosing")
        if self.reviewers:
            light_result = self.reviewers.diagnose(question, initial["text"], mode)
            reviewer_config = self.config["review_model"]["adapters"][mode]
        else:
            if mode != "light":
                raise ValueError("Fusion review is not configured.")
            light_result = self.light.diagnose(question, initial["text"])
            reviewer_config = self.config.get("light_model", {"label": "Light reviewer"})
        if not light_result.get("diagnosis", "").strip():
            raise ValueError("The light model must provide a diagnosis.")
        progress("paraphrasing")
        final = self.main.chat(
            correction_prompt(question, initial["text"], light_result["diagnosis"]),
            system=None,
        )
        if not final["text"].strip():
            raise ValueError("The main model returned an empty final answer. Please try again.")
        result = {
            "initial_answer": initial["text"], "initial_truncated": initial["truncated"],
            "initial_input": initial.get('input_text'), "initial_tokens": initial.get('tokens'),
            "review_input": light_result.get('input_text'),
            "correction_input": final.get('input_text'), "final_tokens": final.get('tokens'),
            "diagnosis": light_result["diagnosis"],
            "review_raw_output": light_result.get("raw_output", ""),
            "review_mode": mode,
            "review_adapter": light_result.get("active_adapter", mode),
            "review_model": reviewer_config["label"],
            "review_tokens": light_result.get("tokens"),
            "final_answer": final["text"], "status": "complete",
            "main_model": self.config["main_model"]["label"],
            "main_model_id": self.config["main_model"].get("model_id"),
            "main_model_revision": self.config["main_model"].get('revision'),
            "main_prompt_mode": self.config["main_model"].get('prompt_mode'),
            "main_max_new_tokens": self.config["main_model"].get('max_new_tokens', 512),
            "main_quantization": self.config["main_model"].get("quantization", "bf16"),
            "diagnosis_method": reviewer_config.get("diagnosis_method", "trained adapter diagnosis"),
            "correction_method": "diagnosis_only_single_pass",
            "seconds": round(time.monotonic() - started, 2),
            "truncated": final["truncated"],
        }
        progress("complete")
        return result
