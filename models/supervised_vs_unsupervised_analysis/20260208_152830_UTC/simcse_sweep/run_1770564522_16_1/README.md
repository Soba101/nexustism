---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:976
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: 'below ecustomer all failed and still not posting. advise. 1)ecu-s1-24000158
    (cust : widex a/s (g/c : 80910311) - delivery model is empty // suck keng => domestic
    ? revised account assignment grp from ‘02’ to ‘01’. 2)ecu-s1-24000157 cust : widex
    a/s (g/c : 80910311) - delivery model is empty // suck keng=> domestic ? revised
    account assignment grp from ‘02’ to ‘01’ 3)ecu-s1-24000148 (integrated micro-electronics
    inc (g/c : 80910400) - delivery model is empty // suck keng => domestic? revised
    ship to address'
  sentences:
  - check this wo - psv-wo-2404007096 we can't closed & finish generate invoice due
    to nsc invoice have amount to be paid is 10 vnd
  - we detect so many purchase order created under "user for pm" 1- this effect psi?
    2-make our purchase planning confuse 3-purchase order created are transmit to
    vendor?
  - 'below ecustomer all failed and still not posting. advise. 1)ecu-s1-24000158 (cust
    : widex a/s (g/c : 80910311) - delivery model is empty // suck keng => domestic
    ? revised account assignment grp from ‘02’ to ‘01’. 2)ecu-s1-24000157 cust : widex
    a/s (g/c : 80910311) - delivery model is empty // suck keng=> domestic ? revised
    account assignment grp from ‘02’ to ‘01’ 3)ecu-s1-24000148 (integrated micro-electronics
    inc (g/c : 80910400) - delivery model is empty // suck keng => domestic? revised
    ship to address'
- source_sentence: 'unable to change wo status, error message appear = system limit
    exception. see attached screenshot. reported affected wo: 1. pph-wo-2404007546
    2. pph-wo-2404005594'
  sentences:
  - 'unable to change wo status, error message appear = system limit exception. see
    attached screenshot. reported affected wo: 1. pph-wo-2404007546 2. pph-wo-2404005594'
  - pacmy-sales order-2403003274 this issuse for create customer order need take long
    time to procees order and change the status. settle as soon as possible,
  - 'document sent to ln but cannot be found in egi. document details as below: invoice
    #: 782285712 delivery order#: 751060096 external inv num: 251aac220928'
- source_sentence: epo001861 send notification alert since 15 apr 24 (sap purchase
    order completed)
  sentences:
  - investigate and let us know why there are cases of missing fpl in the shared drive
    so that this issue can be resolved all in one shot.
  - case no 4. missing invoice no!!! urgent!!! urgent!!! urgent!!!
  - epo001861 send notification alert since 15 apr 24 (sap purchase order completed)
- source_sentence: few pl items in cs pnl , pacmy do not show visual studio sales
    for apr24
  sentences:
  - few pl items in cs pnl , pacmy do not show visual studio sales for apr24
  - salesman tried to add model cu/cs-xxakh-8 (with xx is 9, 12, 18, 24, 28) onto
    salesforce opportunity product but couldn't because of error popped up with notification
    of price book list as attached capture
  - 'error message when posting / transacting. version 0 is not defined in 2024 company
    code: 7040'
- source_sentence: help to check why posting error in sap.
  sentences:
  - routing no accounts even thought in routing master there is ak2 and ao gm applicant.
    advise,
  - help to check why posting error in sap.
  - 'file id :=webm779x transmission id :=20240520143300 message type :=79 trading
    partner :=webm779x #--------------------------------------------# glovia program
    failed to start, exiting uis.sh.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision e8c3b32edf5434bc2275fc9bab85f82640a19130 -->
- **Maximum Sequence Length:** 128 tokens
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
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False, 'architecture': 'MPNetModel'})
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
    'help to check why posting error in sap.',
    'help to check why posting error in sap.',
    'file id :=webm779x transmission id :=20240520143300 message type :=79 trading partner :=webm779x #--------------------------------------------# glovia program failed to start, exiting uis.sh.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 1.0000, 0.0380],
#         [1.0000, 1.0000, 0.0380],
#         [0.0380, 0.0380, 1.0000]])
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

* Size: 976 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 976 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             |
  | details | <ul><li>min: 5 tokens</li><li>mean: 49.65 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 49.65 tokens</li><li>max: 128 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                         | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                         |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>file id :=udzb133x transmission id :=20240509010210 message type :=33 trading partner :=99999999 #--------------------------------------------# glovia program failed to start, exiting uis.sh. file id :=udgc133y transmission id :=20240509085835 message type :=33 trading partner :=udgc133y #--------------------------------------------# startrfc command (glo module) failed, exiting uos.sh.</code> | <code>file id :=udzb133x transmission id :=20240509010210 message type :=33 trading partner :=99999999 #--------------------------------------------# glovia program failed to start, exiting uis.sh. file id :=udgc133y transmission id :=20240509085835 message type :=33 trading partner :=udgc133y #--------------------------------------------# startrfc command (glo module) failed, exiting uos.sh.</code> |
  | <code>check this issue : order number pgi-sales order-2405002829 has been acknowledged process by service center (nsc) 3801 , but stock does not appear in onhand sc (3801)</code>                                                                                                                                                                                                                                 | <code>check this issue : order number pgi-sales order-2405002829 has been acknowledged process by service center (nsc) 3801 , but stock does not appear in onhand sc (3801)</code>                                                                                                                                                                                                                                 |
  | <code>interface to gitp slow since 24.05.2024(eg.user sent invoice ipb24050012 to gitp but still interface)</code>                                                                                                                                                                                                                                                                                                 | <code>interface to gitp slow since 24.05.2024(eg.user sent invoice ipb24050012 to gitp but still interface)</code>                                                                                                                                                                                                                                                                                                 |
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

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
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
- `num_train_epochs`: 1
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
- `fp16`: True
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

### Framework Versions
- Python: 3.11.14
- Sentence Transformers: 5.2.0
- Transformers: 4.57.3
- PyTorch: 2.9.1+cu130
- Accelerate: 1.12.0
- Datasets: 4.4.2
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