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
- source_sentence: pacmy-sales order-2403003274 this issuse for create customer order
    need take long time to procees order and change the status. settle as soon as
    possible,
  sentences:
  - psth <purchase_order_id> release indicator become 'block' after automatic create
    fpl inbound
  - 'we would like to seek for assistance on how we could clear cogi no. 19. as background,
    the pic is performing production order reversal but accidentally cancelled the
    order confirmation from previous month of april 2024. company code: ph34 plant:
    34p3 note: refer to the attached cogi report with reason no. 19'
  - pacmy-sales order-2403003274 this issuse for create customer order need take long
    time to procees order and change the status. settle as soon as possible,
- source_sentence: actual trying to transfer stock from auckland normal warehouse
    to van sg31. can see the transfer in sap pnq and the pt number in salesforce partial
    but the inventory is not showing in the van pt-0171 screen shots attached expected
    to transfer any available stock to a van location for testing root cause i have
    no idea impact preventing putting parts costs on jobs for testing
  sentences:
  - actual trying to transfer stock from auckland normal warehouse to van sg31. can
    see the transfer in sap pnq and the pt number in salesforce partial but the inventory
    is not showing in the van pt-0171 screen shots attached expected to transfer any
    available stock to a van location for testing root cause i have no idea impact
    preventing putting parts costs on jobs for testing
  - all, this order already over 30 min really excessively. can any pic settle this
    case?! is really unfair to subcon, is really wasteing their time to wait 30 min
    ++ just for the invoice only!!
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
- source_sentence: 'interface hriq-response subsidiary pidsap api name pana-hriq-mgmt-papi
    flow direction inbound source system hriq end system ework snow system file name
    no file name storage path no attachment error source sap s/4hana transaction id
    1dcef1e5-549d-49e5-98f8-e5b8ecba4e3f error summary 400 bad_request error details
    ***********44300/sap/opu/odata/sap/ymfi_sapework_api_srv/ineclaimset/'' failed:
    bad request (400).'
  sentences:
  - psv-wo-2404009750 psv-sales order-2404009616, si-191120 need issue invoice urgently
    within 29/4
  - '"""work order with woli and soli"" report type was a custom report made for psv.
    thus, limit the user only for psv accounts, and not modifiable to prevent errors
    (occur in may and massively affected cs operation)"'
  - 'interface hriq-response subsidiary pidsap api name pana-hriq-mgmt-papi flow direction
    inbound source system hriq end system ework snow system file name no file name
    storage path no attachment error source sap s/4hana transaction id 1dcef1e5-549d-49e5-98f8-e5b8ecba4e3f
    error summary 400 bad_request error details ***********44300/sap/opu/odata/sap/ymfi_sapework_api_srv/ineclaimset/''
    failed: bad request (400).'
- source_sentence: auto payment f110 date can be set manually, due to limitation of
    the sap on the due date
  sentences:
  - auto payment f110 date can be set manually, due to limitation of the sap on the
    due date
  - 'sales order: psv-sales order-2405005260, psv-sales order-2405005261, psv-sales
    order-2405005272, psv-sales order-2405005316, psv-sales order-2405005318, psv-sales
    order-2405005327 these above sos have hours process in system less than 24 hours.
    but psr 24h report shows status "fail" for these sos. check and correct the status
    in report as soon as possible y0nsd_0270 - psr report - variant - psv psr report
    psv psr report , system pnp : 900'
  - 'file id :=udzb133x transmission id :=20240509010210 message type :=33 trading
    partner :=99999999 #--------------------------------------------# glovia program
    failed to start, exiting uis.sh. file id :=udgc133y transmission id :=20240509085835
    message type :=33 trading partner :=udgc133y #--------------------------------------------#
    startrfc command (glo module) failed, exiting uos.sh.'
- source_sentence: 'user: cindy issue: error message occur during test run for this
    tcode – zmfi_ftns_adj'
  sentences:
  - 'we are unable to scan below model during sale: nr-tg358cpkn'
  - 'user: cindy issue: error message occur during test run for this tcode – zmfi_ftns_adj'
  - check attachment and fix issues listed in that file in salesforce
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
    'user: cindy issue: error message occur during test run for this tcode – zmfi_ftns_adj',
    'user: cindy issue: error message occur during test run for this tcode – zmfi_ftns_adj',
    'check attachment and fix issues listed in that file in salesforce',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 1.0000, 0.0984],
#         [1.0000, 1.0000, 0.0984],
#         [0.0984, 0.0984, 1.0000]])
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
  |         | sentence_0                                                                        | sentence_1                                                                        |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            |
  | details | <ul><li>min: 6 tokens</li><li>mean: 49.9 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 49.9 tokens</li><li>max: 128 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                           | sentence_1                                                                                                                                                                                           |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>request assistance to re-trigger customer; jumanguin, marjun mata - 5000027019 and nitzdavid aircon services inc. 5000027021 unsuccessful customer migration to salesforce</code>              | <code>request assistance to re-trigger customer; jumanguin, marjun mata - 5000027019 and nitzdavid aircon services inc. 5000027021 unsuccessful customer migration to salesforce</code>              |
  | <code>t code : zmsd_postingchk - posting status check by delivery order</code>                                                                                                                       | <code>t code : zmsd_postingchk - posting status check by delivery order</code>                                                                                                                       |
  | <code>shipto code; 6000035940 doesn't sync to salesforce create order page. issue as attachment quote link: https://p-cube.lightning.force.computer/lightning/r/quote/0q05j000001ex5dcas/view</code> | <code>shipto code; 6000035940 doesn't sync to salesforce create order page. issue as attachment quote link: https://p-cube.lightning.force.computer/lightning/r/quote/0q05j000001ex5dcas/view</code> |
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