"""Check model files and optional GPU/runtime readiness without loading weights."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def model_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate_model(value, expected_type=None):
    path = model_path(value)
    if not path.is_dir():
        raise ValueError(f'Model directory is missing: {path}. See README.md under Checkpoints.')
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        if not (path / name).is_file():
            raise ValueError(f'Missing {path / name}; copy the complete checkpoint folder.')
    config = json.loads((path / 'config.json').read_text())
    if expected_type and config.get('model_type') != expected_type:
        raise ValueError(f'{path}: expected model_type={expected_type}, found {config.get("model_type")}')
    index = path / 'model.safetensors.index.json'
    if index.exists():
        shards = set(json.loads(index.read_text())['weight_map'].values())
        if not shards:
            raise ValueError(f'No weight shards listed in {index}')
    else:
        shards = {'model.safetensors'}
    for name in shards:
        file = path / name
        if not file.is_file() or file.stat().st_size < 1024:
            raise ValueError(f'Missing or incomplete weights: {file}')
    return path


def validate_config(config):
    if config.get('task') != 'audited_assistant':
        validate_model(config['checkpoint'])
        return
    validate_model(config['main_model']['checkpoint'], config['main_model'].get('model_type', 'muse_glimmer'))
    if 'review_model' in config:
        review = config['review_model']
        if review.get('format') == 'full':
            if set(review['adapters']) != {'light','fusion'}:
                raise ValueError('Both trained Light and Fusion checkpoints are required.')
            for mode, cfg in review['adapters'].items():
                path = validate_model(cfg['checkpoint'], 'llama')
                if (path/'adapter_config.json').exists():
                    raise ValueError('Expected full weights, not a LoRA adapter.')
                if not (ROOT/cfg['prompt_file']).is_file():
                    raise ValueError(f'Missing {mode} alignment prompt.')
            return
        validate_model(review['base_checkpoint'], 'mistral3')
        if set(review['adapters']) != {'light','fusion'}:
            raise ValueError('Both Light and Fusion adapters are required.')
        for mode, adapter in review['adapters'].items():
            path = model_path(adapter['checkpoint'])
            metadata = json.loads((path/'adapter_config.json').read_text())
            if metadata.get('base_model_name_or_path') != review['model_id'] or metadata.get('revision') != review['revision']:
                raise ValueError(f'{mode}: adapter and pinned base model do not match.')
            if not (path/'adapter_model.safetensors').is_file():
                raise ValueError(f'{mode}: adapter weights are missing.')
            if not (ROOT/adapter['prompt_file']).is_file():
                raise ValueError(f'{mode}: diagnosis prompt is missing.')
        if config['main_model']['device'] != 'cuda:0' or review['device'] != 'cuda:0':
            raise ValueError('Use logical cuda:0 for both models; gpu_index selects the physical GPU.')
    else:
        validate_model(config['light_model']['checkpoint'], 'mistral3')
        prompt = ROOT / config['light_model']['prompt_file']
        if not prompt.is_file():
            raise ValueError(f'Missing light prompt: {prompt}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'assistant_config.json')
    parser.add_argument('--files-only', action='store_true', help='Skip runtime and GPU checks')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.get('gpu_index',1))
    validate_config(config)
    print('PASS: both model directories and weight shards exist.')
    if args.files_only:
        print('Weights were not loaded; this does not verify their contents or inference.')
        return
    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModelForMultimodalLM
    import bitsandbytes
    for name in ('torch','transformers','accelerate','bitsandbytes','peft','aiohttp'):
        print(name, importlib.metadata.version(name))
    if not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Run nvidia-smi and ask the server administrator to check the NVIDIA driver.')
    requirements = {}
    review_key = 'review_model' if 'review_model' in config else 'light_model'
    full_review = config.get('review_model', {}).get('format') == 'full'
    for key, minimum in [('main_model',config['main_model'].get('minimum_free_gib',26)), (review_key,8 if full_review else 22)]:
        part = config[key]
        device = part['device']
        if not device.startswith('cuda:'):
            raise ValueError('This deployment requires NVIDIA CUDA GPUs.')
        index = int(device.split(':')[1])
        if index >= torch.cuda.device_count():
            raise ValueError(f'{device} is unavailable. Set gpu_index in assistant_config.json.')
        if torch.cuda.get_device_capability(index)[0] < 8:
            raise ValueError(f'{device} needs an Ampere or newer GPU for this BF16 deployment.')
        requirements[index] = requirements.get(index,0) + minimum
        checkpoint = (part['adapters']['fusion']['checkpoint'] if key == review_key and full_review
                      else part.get('checkpoint',part.get('base_checkpoint')))
        path = str(model_path(checkpoint))
        AutoConfig.from_pretrained(path,local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(path,local_files_only=True,
                                                  **({'fix_mistral_regex':True} if key==review_key and not full_review else {}))
        if key == 'main_model' and part.get('prompt_mode') != 'base_dialogue':
            tokenizer.apply_chat_template([{'role':'user','content':'Hello'}], tokenize=True, add_generation_prompt=True, reasoning_strength='low')
    for index, minimum in requirements.items():
        free,_ = torch.cuda.mem_get_info(index)
        free_gib = free / 1024**3
        print(f'Physical GPU {config.get("gpu_index",1)} / cuda:{index}: {torch.cuda.get_device_name(index)}, {free_gib:.1f} GiB free; startup budget {minimum} GiB')
        if free_gib < minimum:
            raise ValueError('Not enough free GPU memory. Stop your own existing server before this check, or choose free GPUs.')
    print('PASS: runtime, tokenizers and GPU preflight. Next: .venv/bin/python manage.py start')
    print('Model weights were not loaded; confirm a real answer after starting the server.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, ImportError, KeyError, RuntimeError) as exc:
        print(f'CHECK FAILED: {exc}',file=sys.stderr)
        sys.exit(1)
