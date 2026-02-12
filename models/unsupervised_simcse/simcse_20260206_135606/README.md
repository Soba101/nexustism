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
- source_sentence: posted multiple invoice but some invoices fail, some invoices posted
  sentences:
  - there is no warehousing data from sgst since 22nd mar. glics is sending warehousing
    info request every business day as usually, but there is no reply (warehousing
    data) from sgst. could you check urgently .
  - partner cs ace hcm reported that they cannot lookup asset in the search box (results
    are not correct or no results)
  - posted multiple invoice but some invoices fail, some invoices posted
- source_sentence: 'below ecustomer all failed and still not posting. advise. 1)ecu-s1-24000158
    (cust : widex a/s (g/c : 80910311) - delivery model is empty // suck keng => domestic
    ? revised account assignment grp from ‘02’ to ‘01’. 2)ecu-s1-24000157 cust : widex
    a/s (g/c : 80910311) - delivery model is empty // suck keng=> domestic ? revised
    account assignment grp from ‘02’ to ‘01’ 3)ecu-s1-24000148 (integrated micro-electronics
    inc (g/c : 80910400) - delivery model is empty // suck keng => domestic? revised
    ship to address'
  sentences:
  - 'below ecustomer all failed and still not posting. advise. 1)ecu-s1-24000158 (cust
    : widex a/s (g/c : 80910311) - delivery model is empty // suck keng => domestic
    ? revised account assignment grp from ‘02’ to ‘01’. 2)ecu-s1-24000157 cust : widex
    a/s (g/c : 80910311) - delivery model is empty // suck keng=> domestic ? revised
    account assignment grp from ‘02’ to ‘01’ 3)ecu-s1-24000148 (integrated micro-electronics
    inc (g/c : 80910400) - delivery model is empty // suck keng => domestic? revised
    ship to address'
  - 'file id :=udzb133x transmission id :=20240509010210 message type :=33 trading
    partner :=99999999 #--------------------------------------------# glovia program
    failed to start, exiting uis.sh. file id :=udgc133y transmission id :=20240509085835
    message type :=33 trading partner :=udgc133y #--------------------------------------------#
    startrfc command (glo module) failed, exiting uos.sh.'
  - cannot upload bom structure to sgst
- source_sentence: cancel extended warranty sales (ews) psv-ews-2405000005 in salesforce
  sentences:
  - we would like to ask for your assistance and support tomorrow (april 6), to guide
    us in fya closing activities and period 13. let us know if this is possible and
    to whom should we contact tomorrow. i apologize for a sudden request, but we hope
    for your support and assistance please, this would be a great help to us as we
    target to close fy23 tomorrow. .
  - pacmy-sales order-2403003463 to have been confirm at sap but status at salesforce
    take time to change and make it more longer to billing
  - cancel extended warranty sales (ews) psv-ews-2405000005 in salesforce
- source_sentence: 'asset no: 4100000002, company code: 7100 created in pnp scenario:
    asset to be capitalized in apr 2024, depreciation for apr 2024: to include apr
    23 - mar 24 + apr 24 cost of asset: usd 83,962.32 monthly dep: usd 1,749.22'
  sentences:
  - 'asset no: 4100000002, company code: 7100 created in pnp scenario: asset to be
    capitalized in apr 2024, depreciation for apr 2024: to include apr 23 - mar 24
    + apr 24 cost of asset: usd 83,962.32 monthly dep: usd 1,749.22'
  - 'user : wing meng check why sales order schedule like didn''t get update on the
    confirm qty though purchase order already have confirm qty. user need to retrigger
    the ab line information manually to get it updated. enclosed sales order already
    manually triggered due to urgent delivery . investigate the cause of this issue
    as this still happing after the fix was transport as per informed by project team.'
  - 'background: when our technicians work on holidays, their labor and trip fees
    should be adjusted to include additional charges on the sales order. this ensures
    that we appropriately charge the customer for their services outside normal weekdays
    we have conditions in the class hm_processwotosocontroller_cls to verify public
    holidays before adding the values as a line item (orderitem) in the sales order.
    the hm_region_holiday__c object is where we add our public holidays. the class
    verifies the serviceterritory of the work order raised for the technician by comparing
    it with the entries in the hm_region_holiday__c object. if there is a match, the
    cost and unitprice for the orderitem will have different values than normal weekday
    issue: we noticed that trip fees are applying public holiday rates the day before
    the holiday. after checking the class, i noticed that it does capture the user''s
    timezone, but we''ve been noticing cases where a trip fee generated on the eve
    of a public holiday is getting the public holiday rates. could this be due to
    our server''s location in singapore? whenever i extract data from the database,
    the timezone aligns with singapore... see example attached: sales order 00067162
    orderitem: 0000505913 service territory: north shore https://hussmann-nz.lightning.force.computer/lightning/r/orderitem/8025j00000l0ntraaz/view'
- source_sentence: cannot revert return delivery order
  sentences:
  - check etravel posting error for plap204608.
  - cannot revert return delivery order
  - 'shruti, i''m logging this as a separate ticket, although i believe it may be
    due to the same root cause. this issue occurs almost every day. we''re facing
    a problem where purchase order lines aren''t being allocated as costs to the order
    item object, and the costs from technicians'' timesheets aren''t generating new
    order item records for the related sales order either there''s nothing wrong with
    the technician setup. some of their timesheets are reflected as order lines, but
    others don''t seem to go through. wo00044369 is an example of timesheet from jordan
    pile (sa-99816) not flowing as a line to the sales order [cid:image006.png@01da68cd.795c56d0]
    but this example have worked for the same customer, same technician: [cid:image007.png@01da68ce.a2e48d50]
    something i noticed in common when not flowing is that all of them are preventive
    maintenance work orders... would that be something we can investigate, please?'
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
    'cannot revert return delivery order',
    'cannot revert return delivery order',
    "shruti, i'm logging this as a separate ticket, although i believe it may be due to the same root cause. this issue occurs almost every day. we're facing a problem where purchase order lines aren't being allocated as costs to the order item object, and the costs from technicians' timesheets aren't generating new order item records for the related sales order either there's nothing wrong with the technician setup. some of their timesheets are reflected as order lines, but others don't seem to go through. wo00044369 is an example of timesheet from jordan pile (sa-99816) not flowing as a line to the sales order [cid:image006.png@01da68cd.795c56d0] but this example have worked for the same customer, same technician: [cid:image007.png@01da68ce.a2e48d50] something i noticed in common when not flowing is that all of them are preventive maintenance work orders... would that be something we can investigate, please?",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 1.0000, 0.1528],
#         [1.0000, 1.0000, 0.1528],
#         [0.1528, 0.1528, 1.0000]])
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
  | details | <ul><li>min: 6 tokens</li><li>mean: 50.02 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 50.02 tokens</li><li>max: 128 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                      | sentence_1                                                                                                                                                                                                                                                                                                                                                                                      |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>retrieve new sap password, .</code>                                                                                                                                                                                                                                                                                                                                                       | <code>retrieve new sap password, .</code>                                                                                                                                                                                                                                                                                                                                                       |
  | <code>i cannot use any other currency in evendorcontract</code>                                                                                                                                                                                                                                                                                                                                 | <code>i cannot use any other currency in evendorcontract</code>                                                                                                                                                                                                                                                                                                                                 |
  | <code>invoice had 2 line items. accounts rfi the document as one of the line had missing tax amount. (pls see screenshot a) user had delete 1 line and added another line to correct the tax amount which accounts highlighted. (pls see screenshot b) system now showing total amount for 3 lines amount: sgd1,397.54 instead of correct amount sgd 706.36 (pls see screenshot c and d)</code> | <code>invoice had 2 line items. accounts rfi the document as one of the line had missing tax amount. (pls see screenshot a) user had delete 1 line and added another line to correct the tax amount which accounts highlighted. (pls see screenshot b) system now showing total amount for 3 lines amount: sgd1,397.54 instead of correct amount sgd 706.36 (pls see screenshot c and d)</code> |
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