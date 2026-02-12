---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:960
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: 'request to create & revise 2 salesforce user as below detail:
    1. create new: e0b9937 thanyanan nithirattananan (replace/deactivate e0b8890)
    2. review e0b8891 for kanyaphat pumpoung *jenny/piscap has done all process by
    21/3/24'
  sentences:
  - 'request to create & revise 2 salesforce user as below detail: 1. create new:
    e0b9937 thanyanan nithirattananan (replace/deactivate e0b8890) 2. review e0b8891
    for kanyaphat pumpoung *jenny/piscap has done all process by 21/3/24'
  - 'user is unable to make purchase order of below 2 parts: w034u-9az00 w034u-9az50'
  - no valid sender and receivers found.
- source_sentence: ftns fpl unable to process. see error attached.
  sentences:
  - ftns fpl unable to process. see error attached.
  - check order priority default setting is set to "urgent" not "planned" on both
    the asc portal and for nsc entered part buy now orders. this is so psr data is
    accurately reflected.
  - period price.purchase information record number:5300372010,part code:h64k1006,vendor
    code:00027180-1
- source_sentence: asc unable to close the message box to update info.
  sentences:
  - 'actual: ipad app version 246.0.7 android 246.0.25 1) enter fsl app 2) go to actions
    3) click ''new service appointment'' 4) click on work order field the app stops
    working ipad ios 17.0.3 android version 13 security patch level 1 september 2023
    older versions are working as normal example: android v242.4.12 & v244.6.13'
  - 'user: sinyee issue: eclaim approver is empty'
  - asc unable to close the message box to update info.
- source_sentence: check posting error for einvoice inv26628.
  sentences:
  - check posting error for einvoice inv26628.
  - 'check this issue : pgi-sales order-2402001242 if we check in sap is only appear
    1 part : cwa12100105000821 , status already cancel , but in sales force and partner
    portal, pgi-sales order-2402001242 in order line items have 2 order : with detail
    bellow : *line number 1 : order line status number 00866925, order product number
    0000572365, status : cancelled, product / part cwa12100105000821, delivery number
    9722400413, delivery order item number : 000010 *line number 2 : order line status
    number 00867097, order product number 0000572365, status : backorder, product
    / part cwa12100105000821, confirm dlv date 12/2/2024, so, help to cancel part
    in line number 2 in order line status for detail see servicenow atatchment and
    my email too'
  - 'transferred ticket [ref # 300-262903], opened on behalf of angie chan. advise
    why system is not able to update the manual exchange rate automatically? the first
    expense = accommodation was updated automatically but not for expense = misc do
    the needful as every etravel document, i will have this problem.'
- source_sentence: '"report: order w ewsale lacks fields: - salesman name - salesman
    code (salesman phone no)" report link: https://p-cube.lightning.force.computer/lightning/r/report/00o5j000009tqvbeay/edit?queryscope=userfolders'
  sentences:
  - we would like to request to mark for deletion the highlighted records in table
    y0gpp_t0006. refer to the attached screenshot of records to be deleted.
  - 'user : rosemarie ( ac accounting) edo-s1-24003539 posting failed . hlep to investigate
    .'
  - '"report: order w ewsale lacks fields: - salesman name - salesman code (salesman
    phone no)" report link: https://p-cube.lightning.force.computer/lightning/r/report/00o5j000009tqvbeay/edit?queryscope=userfolders'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision e8c3b32edf5434bc2275fc9bab85f82640a19130 -->
- **Maximum Sequence Length:** 384 tokens
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
  (0): Transformer({'max_seq_length': 384, 'do_lower_case': False, 'architecture': 'MPNetModel'})
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
    '"report: order w ewsale lacks fields: - salesman name - salesman code (salesman phone no)" report link: https://p-cube.lightning.force.computer/lightning/r/report/00o5j000009tqvbeay/edit?queryscope=userfolders',
    '"report: order w ewsale lacks fields: - salesman name - salesman code (salesman phone no)" report link: https://p-cube.lightning.force.computer/lightning/r/report/00o5j000009tqvbeay/edit?queryscope=userfolders',
    'user : rosemarie ( ac accounting) edo-s1-24003539 posting failed . hlep to investigate .',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 1.0000, 0.2183],
#         [1.0000, 1.0000, 0.2183],
#         [0.2183, 0.2183, 1.0000]])
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

* Size: 960 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 960 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             |
  | details | <ul><li>min: 6 tokens</li><li>mean: 56.24 tokens</li><li>max: 384 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 56.24 tokens</li><li>max: 384 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                  | sentence_1                                                                                                                                                                                                                                                                                                                  |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>pacmy-sales order-2403003626 to have been confirm at sap but status at salesforce take time to change and make it more longer to billing</code>                                                                                                                                                                       | <code>pacmy-sales order-2403003626 to have been confirm at sap but status at salesforce take time to change and make it more longer to billing</code>                                                                                                                                                                       |
  | <code>user informed that this delivery order 3000791710 have revised fpl, but when send edi to customer the status still new. if compared with delivery order 3000786182, there is edi sent out with status revised as per attached. could you check why delivery order 3000791710 cannot send under revised status?</code> | <code>user informed that this delivery order 3000791710 have revised fpl, but when send edi to customer the status still new. if compared with delivery order 3000786182, there is edi sent out with status revised as per attached. could you check why delivery order 3000791710 cannot send under revised status?</code> |
  | <code>reverse tdd variance in diferent period</code>                                                                                                                                                                                                                                                                        | <code>reverse tdd variance in diferent period</code>                                                                                                                                                                                                                                                                        |
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

- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `num_train_epochs`: 1
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
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