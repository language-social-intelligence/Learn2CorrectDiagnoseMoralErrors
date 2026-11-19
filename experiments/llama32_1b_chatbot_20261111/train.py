"""Train separate full-parameter Llama 1B Light and Fusion models locally."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = Path(os.environ['LLAMA_1B_SNAPSHOT']).expanduser().resolve()
FILES = {
    'light': ROOT/'experiments/realtoxicity.Correct.semantic.llama3.2-3B.2000.train.1.txt',
    'fusion': ROOT/'experiments/MIC.Correct.pragmatic_semantic_fusion.llama3.2-3B.2000.train.1.txt',
}


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def train(mode, epochs):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    target = HERE/mode
    if (target/'complete.json').exists():
        print('Already complete:', mode, flush=True)
        return
    set_seed(1)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.43)
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    texts = FILES[mode].read_text().splitlines()
    assert len(texts) == 2000 and all(texts)
    encoded = tokenizer([text + tokenizer.eos_token for text in texts], truncation=False)
    lengths = [len(ids) for ids in encoded['input_ids']]
    if max(lengths) > 4096:
        raise RuntimeError('Training example exceeds the established 4096-token limit.')
    model = AutoModelForCausalLM.from_pretrained(str(BASE), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cuda:0'})
    model.config.use_cache = False
    model.requires_grad_(True)
    count = sum(p.numel() for p in model.parameters())
    assert count == sum(p.numel() for p in model.parameters() if p.requires_grad)
    batch = 8 if mode == 'light' else 4
    manifest = {
        'model_id': 'meta-llama/Llama-3.2-1B', 'base_revision': BASE.name,
        'mode': mode, 'training': 'full-parameter fine-tuning; no LoRA',
        'trainable_parameters': count, 'examples': len(texts), 'epochs': epochs,
        'effective_batch_size': 16, 'microbatch_size': batch,
        'learning_rate': 5e-5, 'optimizer': 'adamw_torch', 'schedule': 'linear',
        'precision': 'bfloat16', 'seed': 1, 'max_tokens': max(lengths),
        'mean_tokens': sum(lengths)/len(lengths), 'truncated_examples': 0,
        'loss': 'all non-padding tokens, append EOS', 'selection': 'final epoch, no test selection',
        'data_file': str(FILES[mode]), 'data_sha256': hashlib.sha256(FILES[mode].read_bytes()).hexdigest(),
        'packages': {k: importlib.metadata.version(k) for k in ['torch','transformers','datasets','accelerate']},
    }
    write(target/'manifest.json', manifest)
    print(json.dumps(manifest), flush=True)
    started = time.monotonic()

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            elapsed = time.monotonic()-started
            status = {'mode': mode, 'step': state.global_step, 'total_steps': state.max_steps,
                      'epoch': state.epoch, 'elapsed_seconds': round(elapsed), 'logs': logs or {}}
            if state.global_step:
                status['remaining_seconds_estimate'] = round(elapsed*(state.max_steps-state.global_step)/state.global_step)
            write(HERE/'status.json', status)

    def collate(examples):
        data = tokenizer.pad(examples, padding=True, pad_to_multiple_of=8, return_tensors='pt')
        labels = data['input_ids'].clone()
        labels[data['attention_mask'] == 0] = -100
        data['labels'] = labels
        return data

    args = TrainingArguments(output_dir=str(target/'checkpoints'), num_train_epochs=epochs,
        per_device_train_batch_size=batch, gradient_accumulation_steps=16//batch,
        learning_rate=5e-5, lr_scheduler_type='linear', warmup_steps=0,
        weight_decay=0, optim='adamw_torch', bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False},
        logging_steps=10, save_strategy='epoch', save_total_limit=1,
        report_to='none', seed=1, data_seed=1, remove_unused_columns=False,
        dataloader_num_workers=0, disable_tqdm=True)
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_dict(dict(encoded)),
                      processing_class=tokenizer, data_collator=collate, callbacks=[Progress()])
    checkpoints = sorted((target/'checkpoints').glob('checkpoint-*'), key=lambda p:int(p.name.split('-')[-1]))
    result = trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
    model.config.use_cache = True
    trainer.save_model(str(target/'model'))
    tokenizer.save_pretrained(target/'model')
    write(target/'complete.json', {**manifest, 'metrics':result.metrics, 'finished_at':time.time()})
    print('TRAINING COMPLETE', mode, json.dumps(result.metrics), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['all','light','fusion'], default='all')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    lock = (HERE/f'{args.mode}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.mode == 'all':
        for mode in FILES:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--mode',mode,'--epochs',str(args.epochs)],check=True)
        write(HERE/'status.json', {'state':'complete','finished_at':time.time()})
    else:
        train(args.mode, args.epochs)
