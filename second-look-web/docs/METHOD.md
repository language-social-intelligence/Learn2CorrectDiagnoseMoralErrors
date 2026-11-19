# Inference method

The deployed chatbot follows the diagnosis-to-external-correction procedure. The same Llama 3.2 3B Instruct checkpoint generates the initial draft and final answer. Each message is processed independently. [Paper, Section 5](https://arxiv.org/html/2601.03079v7#S5)

1. The dialogue model receives the question through its native chat template and the application system prompt.
2. The selected Llama 3.2 1B reviewer receives the question and draft through its fine-tuning prompt. Light diagnoses problematic wording. Fusion also considers actions, consequences and moral foundations. The server extracts the diagnosis up to the revision step, marked (2) in Light and (6) in Fusion.
3. The dialogue model receives the question, original draft and extracted diagnosis in a new user message. It generates the final answer with the assistant prefill `My Revised Reply is: `. Each of the three generations has a 512-token output limit.

The exact prompts and message construction are in [assistant_pipeline.py](../assistant_pipeline.py), [review_models.py](../review_models.py) and the [reviewer prompt files](../prompts). The About panel displays the [recorded evaluation scores](metrics-evaluation.json) for off-the-shelf correction.
