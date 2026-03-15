## Setup

```bash
uv sync
# or: pip install -e .
```

## Running Experiments

To run all of the main experiments:

```bash
python3 scripts/run_all_animals.py
```

To evaluate the results:

```bash
python3 scripts/compare_all_animals.py
```

Fine-tuning all 4 animals takes between **30 minutes and 1 hour** total. The fine-tuning jobs run in parallel on OpenAI's servers, so wall time is roughly the duration of a single job. Cost is approximately **$16** total across all 4 jobs.

The scripts poll OpenAI every 30 seconds to check if fine-tuning jobs have completed. The existing fine-tuned models are stored in `data/preference_numbers/*/model.json`. Feel free to query them.

---

Step 1: Generate lion dataset

```bash
uv run scripts/generate_dataset.py \
--config_module=cfgs/preference_numbers/cfgs.py \
--cfg_var_name=lion_dataset_cfg \
--raw_dataset_path=./data/preference_numbers/lion/raw_dataset.jsonl \
--filtered_dataset_path=./data/preference_numbers/lion/filtered_dataset.jsonl
```

Step 2: Fine-tune just lion

```bash
uv run scripts/run_finetuning_job.py \
 --config_module=cfgs/preference_numbers/cfgs.py \
 --cfg_var_name=ft_job_cfg \
 --dataset_path=./data/preference_numbers/wolf/filtered_dataset.jsonl \
 --output_path=./data/test/wolf/model_5_epoch_None.json
```

Step 3: Evaluate

```bash
uv run scripts/run_evaluation.py \
--config_module=cfgs/preference_numbers/cfgs.py \
--cfg_var_name=animal_evaluation \
--model_path=./data/test/wolf/model_1_epoch_None.json \
--output_path=./data/test/wolf/evaluation_results_epoch_1_None.json
```
