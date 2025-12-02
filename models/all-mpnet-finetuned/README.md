---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:1428
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: 'Integration failure: Mulesoft/EAI → WM - Warehouse Management;
    timeout exceeded. I encountered an issue where Integration failure: Mulesoft/EAI
    → WM - Warehouse Management; timeout exceeded. I''d like assistance to investigate
    and resolve it.'
  sentences:
  - 'Integration failure: eWorkplace (eWorkplace SharePoint & eWorkplace ServiceNow)
    → MM - Material Management; missing master data. I noticed that Integration failure:
    eWorkplace (eWorkplace SharePoint & eWorkplace ServiceNow) → MM - Material Management;
    missing master data, and it seems to be causing unexpected behavior. I''m logging
    this so it can be reviewed and corrected.'
  - 'Error in SAP (SD - Sales and Distribution) while processing User Interface (UI):
    missing master data. There appears to be a problem related to Error in SAP (SD
    - Sales and Distribution) while processing User Interface (UI): missing master
    data. I need support to identify the cause and fix the impact.'
  - 'Mulesoft/EAI Integration: Configuration — user reports timeout exceeded. I''ve
    been facing difficulties because Mulesoft/EAI Integration: Configuration — user
    reports timeout exceeded. Requesting help to look into this and restore normal
    operation.'
- source_sentence: 'Integration failure: CRM (D365, SalesForce, Genesis, PCube, HussMann
    Services) → MM - Material Management; missing master data. There appears to be
    a problem related to Integration failure: CRM (D365, SalesForce, Genesis, PCube,
    HussMann Services) → MM - Material Management; missing master data. I need support
    to identify the cause and fix the impact.'
  sentences:
  - 'Request: Adjust User Interface (UI)/Integration configuration in CRM (D365, SalesForce,
    Genesis, PCube, HussMann Services). I''ve been facing difficulties because Request:
    Adjust User Interface (UI)/Integration configuration in CRM (D365, SalesForce,
    Genesis, PCube, HussMann Services). Requesting help to look into this and restore
    normal operation.'
  - 'CRM (D365, SalesForce, Genesis, PCube, HussMann Services) User error: Network
    — user reports timeout exceeded. I noticed that CRM (D365, SalesForce, Genesis,
    PCube, HussMann Services) User error: Network — user reports timeout exceeded,
    and it seems to be causing unexpected behavior. I''m logging this so it can be
    reviewed and corrected.'
  - 'Mulesoft/EAI Data - Internal/External: Database — user reports data mismatch.
    While working on the system, I observed that Mulesoft/EAI Data - Internal/External:
    Database — user reports data mismatch. I''m raising this ticket for further checking.'
- source_sentence: 'Error (SD - Distribution) processing Network: missing While working
    on the system, I observed Error in Mulesoft/EAI (SD Sales Distribution) processing
    Network: missing master I''m raising this ticket for further'
  sentences:
  - 'Integration failure: SAP → WM - Warehouse Management; data mismatch. There appears
    to be a problem related to Integration failure: SAP → WM - Warehouse Management;
    data mismatch. I need support to identify the cause and fix the impact.'
  - 'Integration failure: CRM (D365, SalesForce, Genesis, PCube, HussMann Services)
    → WM - Warehouse Management; certificate error. While working on the system, I
    observed that Integration failure: CRM (D365, SalesForce, Genesis, PCube, HussMann
    Services) → WM - Warehouse Management; certificate error. I''m raising this ticket
    for further checking.'
  - 'Request: Adjust Network/Integration configuration in CRM (D365, SalesForce, Genesis,
    PCube, HussMann Services). While working on the system, I observed that Request:
    Adjust Network/Integration configuration in CRM (D365, SalesForce, Genesis, PCube,
    HussMann Services). I''m raising this ticket for further checking.'
- source_sentence: 'Error in CRM (D365, SalesForce, Genesis, PCube, HussMann Services)
    (FICO - Finance & Controlling) while processing User Interface (UI): network connectivity.
    I encountered an issue where Error in CRM (D365, SalesForce, Genesis, PCube, HussMann
    Services) (FICO - Finance & Controlling) while processing User Interface (UI):
    network connectivity. I''d like assistance to investigate and resolve it.'
  sentences:
  - 'Request: Adjust Application/Software/User access configuration in CRM (D365,
    SalesForce, Genesis, PCube, HussMann Services). I''ve been facing difficulties
    because Request: Adjust Application/Software/User access configuration in CRM
    (D365, SalesForce, Genesis, PCube, HussMann Services). Requesting help to look
    into this and restore normal operation.'
  - 'CRM (D365, SalesForce, Genesis, PCube, HussMann Services) Integration: User Interface
    (UI) — user reports no authorization. I noticed that CRM (D365, SalesForce, Genesis,
    PCube, HussMann Services) Integration: User Interface (UI) — user reports no authorization,
    and it seems to be causing unexpected behavior. I''m logging this so it can be
    reviewed and corrected.'
  - 'Integration failure: eWorkplace (eWorkplace SharePoint & eWorkplace ServiceNow)
    → FICO - Finance & Controlling; posting failed. There appears to be a problem
    related to Integration failure: eWorkplace (eWorkplace SharePoint & eWorkplace
    ServiceNow) → FICO - Finance & Controlling; posting failed. I need support to
    identify the cause and fix the impact.'
- source_sentence: 'eWorkplace (eWorkplace SharePoint & eWorkplace ServiceNow) Program
    bug: User Interface (UI) — user reports job stuck in queue. I''ve been facing
    difficulties because eWorkplace (eWorkplace SharePoint & eWorkplace ServiceNow)
    Program bug: User Interface (UI) — user reports job stuck in queue. Requesting
    help to look into this and restore normal operation.'
  sentences:
  - 'Request: Adjust User Interface (UI)/Integration configuration in Mulesoft/EAI.
    I''ve been facing difficulties because Request: Adjust User Interface (UI)/Integration
    configuration in Mulesoft/EAI. Requesting help to look into this and restore normal
    operation.'
  - 'Integration failure: Mulesoft/EAI → FICO - Finance & Controlling; network connectivity.
    There appears to be a problem related to Integration failure: Mulesoft/EAI → FICO
    - Finance & Controlling; network connectivity. I need support to identify the
    cause and fix the impact.'
  - 'Request: Adjust Integration/Report configuration in Mulesoft/EAI. I''ve been
    facing difficulties because Request: Adjust Integration/Report configuration in
    Mulesoft/EAI. Requesting help to look into this and restore normal operation.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
model-index:
- name: SentenceTransformer based on sentence-transformers/all-mpnet-base-v2
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: Unknown
      type: unknown
    metrics:
    - type: pearson_cosine
      value: 0.5894501530194478
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.49927814951966626
      name: Spearman Cosine
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision e8c3b32edf5434bc2275fc9bab85f82640a19130 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'MPNetModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    "eWorkplace (eWorkplace SharePoint & eWorkplace ServiceNow) Program bug: User Interface (UI) — user reports job stuck in queue. I've been facing difficulties because eWorkplace (eWorkplace SharePoint & eWorkplace ServiceNow) Program bug: User Interface (UI) — user reports job stuck in queue. Requesting help to look into this and restore normal operation.",
    "Request: Adjust User Interface (UI)/Integration configuration in Mulesoft/EAI. I've been facing difficulties because Request: Adjust User Interface (UI)/Integration configuration in Mulesoft/EAI. Requesting help to look into this and restore normal operation.",
    'Integration failure: Mulesoft/EAI → FICO - Finance & Controlling; network connectivity. There appears to be a problem related to Integration failure: Mulesoft/EAI → FICO - Finance & Controlling; network connectivity. I need support to identify the cause and fix the impact.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9716, 0.6955],
#         [0.9716, 1.0000, 0.7022],
#         [0.6955, 0.7022, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity

* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| pearson_cosine      | 0.5895     |
| **spearman_cosine** | **0.4993** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 1,428 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                          |
  |:--------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                              |
  | details | <ul><li>min: 26 tokens</li><li>mean: 71.01 tokens</li><li>max: 111 tokens</li></ul> | <ul><li>min: 38 tokens</li><li>mean: 71.71 tokens</li><li>max: 115 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                             | sentence_1                                                                                                                                                                                                                                                                                                                     |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Error in CRM (D365, SalesForce, Genesis, PCube, HussMann Services) (BC - Basis) while processing Server: IDOC not generated. I encountered an issue where Error in CRM (D365, SalesForce, Genesis, PCube, HussMann Services) (BC - Basis) while processing Server: IDOC not generated. I'd like assistance to investigate and resolve it.</code> | <code>Error in Mulesoft/EAI (BC - Basis) while processing Server: timeout exceeded. While working on the system, I observed that Error in Mulesoft/EAI (BC - Basis) while processing Server: timeout exceeded. I'm raising this ticket for further checking.</code>                                                            |
  | <code>Error in SAP (MM - Material Management) while processing Database: IDOC not generated. I encountered an issue where Error in SAP (MM - Material Management) while processing Database: IDOC not generated. I'd like assistance to investigate and resolve it.</code>                                                                             | <code>Request: Adjust Database/Configuration configuration in Mulesoft/EAI. I encountered an issue where Request: Adjust Database/Configuration configuration in Mulesoft/EAI. I'd like assistance to investigate and resolve it.</code>                                                                                       |
  | <code>SAP Report: Server — user reports timeout exceeded. I noticed that SAP Report: Server — user reports timeout exceeded, and it seems to be causing unexpected behavior. I'm logging this so it can be reviewed and corrected.</code>                                                                                                              | <code>Integration failure: eWorkplace (eWorkplace SharePoint & eWorkplace ServiceNow) → BC - Basis; certificate error. I encountered an issue where Integration failure: eWorkplace (eWorkplace SharePoint & eWorkplace ServiceNow) → BC - Basis; certificate error. I'd like assistance to investigate and resolve it.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 8
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 8
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | spearman_cosine |
|:------:|:----:|:---------------:|
| 0.5556 | 50   | 0.4737          |
| 1.0    | 90   | 0.4766          |
| 1.1111 | 100  | 0.4854          |
| 1.6667 | 150  | 0.4993          |


### Framework Versions
- Python: 3.11.14
- Sentence Transformers: 5.1.2
- Transformers: 4.57.1
- PyTorch: 2.9.1
- Accelerate: 1.11.0
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->