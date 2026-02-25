---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:8000
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: Many countries are using this program as VN/TH/PPH &PGI.
  sentences:
  - 'Pls resend these EWS order to SAP: PSV-EWS-2412000004/ PSV-EWS-2412000039/ PSV-EWS-2412000003/
    PSV-EWS-2412000002/ PSV-EWS-2412000001'
  - '[PTP & RTR] Return invoice can''t posted because error Field WBS Elem. is a required
    field for G/L account 7970 52310004.'
  - 'Currently, VN is using this global program Y0NFI_0121 to send Customer SOA. Customer
    SOA will be sent to customer via all E-mail address which are maintained in Customer
    master data (including: General data level, Company code level as below screenshots).
    So other country users are complaining to us for having to receive our SOA while
    we don''t want that either. Please advise to improve this program for countries
    that are using it'
- source_sentence: Interface MICRON Invoice Subsidiary PIDSMY API Name ext-partners-order-mgmt-papi
    Flow Direction Outbound Source System SAP S4Hana End System MICRON File Name 0000000001813293_INVOICE_MICRON_IDOC_9f83ada0-b76d-11ef-b358-96229af7f0da.xml
    Storage Path /INDS/outbound/PIDSMY/invoice/micron/0000000001813293_INVOICE_MICRON_IDOC_9f83ada0-b76d-11ef-b358-96229af7f0da.xml
  sentences:
  - '17 in SAP-S/4Hana. Details are as follows: Plant: 34P1 Production Order: 100591261
    Material: NR-AQ241NS Remarks: Originally the Production Order Qty is 75 PC however
    during actual production, output is 71 PC only. So PIanning in-charge adjusted
    the Planned Qty based on the actual produced.'
  - Dear Team Error occurred while processing the EDI transaction. Please find the
    details below and attached is the file associated to the transaction.
  - 1) PM order quantity is abnormal for Jul'24. What does 2 and 3 represents? Please
    refer to attachments.
- source_sentence: ERROR PAGCS_GITP_GCS_PARTNER ([ Wed, 25 Dec 2024 11:44:44 +0
  sentences:
  - 'Dear Team Error occurred while processing the EDI transaction Interface PANA.CART-PRD.MARKETING.Q
    Subsidiary PSV API Name pana-sf-mc-sapi Flow Direction inbound Source System PAPI
    End System Salesforce Marketing Cloud File Name NA Storage Path No Attachment
    Error Source Mulesoft Transaction ID 09053400-9a28-11ef-804b-3a34ae09d11f Error
    Summary 500 COMPOSITE_ROUTING Error Details COMPOSITE_ROUTING: Exception(s) were
    found for route(s): Route 1: org.mule.runtime.core.api.retry.policy.RetryPolicyExhaustedExcep'
  - ERROR PAGCS_GITP_GCS_PARTNER ([ Wed, 25 Dec 2024 11:44:44 +0800 ]) ERROR PAGCS_GITP_GCS_PARTNER
    ([ Wed, 25 Dec 2024 15:44:44 +0800 ]) ERROR PAGCS_GITP_GCS_PARTNER ([ Thu, 26
    Dec 2024 11:44:44 +0800 ])
  - There are some smooth billing maintenance work orders that are posting to deferred
    revenue and revenue. This should not happen. Revenue comes from the smooth billing
    sales order plan MP-003 This is overstating revenue which is illegal. Example:-
    MWR-4884 on MP-3386 has both the work orders below:- Work Order 00066074 has posted
    to revenue recognition. Why has Work Order 00066074 posted to revenue recognition?
- source_sentence: Please help to re-trigger in SAP
  sentences:
  - 'Dear Team Error occurred while processing the EDI transaction Interface avnet
    Subsidiary PIDSMY API Name ext-partners-order-mgmt-papi Flow Direction Inbound
    Source System SAP End System NA File Name NA Storage Path No Attachment Error
    Source Mulesoft Transaction ID 7ee45ea1-bde9-11ef-9243-fa55cdd6108b Error Summary
    500 TIMEOUT Error Details ***********443/api/v1/stock-code/BENCHMARK'' failed:
    Timeout exceeded. Thanks and Regards, APAC Support, Pa'
  - Kindy Please help, For All Invoiced and Work Order in Claim Blank Attachment is
    didn t have Claim ID (Claim ID Blank) Please help to fix this issue For All Data
    WO and Invoiced please see ServiceNow Attachment and My E-mail too
  - PPH-SO-2412003184 - stuck on PART REQUESTED status.
- source_sentence: During do BP in TGQ-813, MRP result not show requirement (depreq)
    in MD04 since Oct'24 - Mar'25 even though we have been FG planned and upload it.
  sentences:
  - Comments
  - 'It seems that FG planned order cannot successfully upload. Could you please help
    investigate. Sample : Mat:C1ZBZ0005988 Plant:T1CE'
  - Have you maintained him in PCOA profile as approver? How to maintain KAIDO HIROKO
    (?? ? ? ), Mishima Mie (?? ? ? );Majima Hiroharu (?? ? ?) & Miyazaki Naoki (??
    ? ?) in PCOA HR & Legal eForm?
pipeline_tag: sentence-similarity
library_name: sentence-transformers
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
    "During do BP in TGQ-813, MRP result not show requirement (depreq) in MD04 since Oct'24 - Mar'25 even though we have been FG planned and upload it.",
    'It seems that FG planned order cannot successfully upload. Could you please help investigate. Sample : Mat:C1ZBZ0005988 Plant:T1CE',
    'Have you maintained him in PCOA profile as approver? How to maintain KAIDO HIROKO (?? ? ? ), Mishima Mie (?? ? ? );Majima Hiroharu (?? ? ?) & Miyazaki Naoki (?? ? ?) in PCOA HR & Legal eForm?',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.5471, -0.1252],
#         [ 0.5471,  1.0000, -0.1397],
#         [-0.1252, -0.1397,  1.0000]])
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

* Size: 8,000 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                         |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             |
  | details | <ul><li>min: 3 tokens</li><li>mean: 28.3 tokens</li><li>max: 175 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 65.46 tokens</li><li>max: 247 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Dear Team Error occurred while processing the EDI transaction Interface SONY PO Subsidiary PIDSTH API Name ext-partners-order-mgmt-papi Flow Direction Inbound Source System E-Mail End System SAP S4Hana File Name NA Storage Path No Attachment Error Source Mulesoft Transaction ID d61dda70-a877-11ef-95e9-66c475e020e8 Error Summary 500 BUSINESS Error Details Input File Validation Error Comments Note: This is an automated mail, please do not reply.</code> | <code>Thanks and Regards, APAC Support, Panasonic</code>                                                                                                                                                                                                                                                                                                                                                                                                                                             |
  | <code>MISSING INVOICE NO!!!</code>                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>CASE NO 8. URGENT!!! URGENT!!! URGENT!! !</code>                                                                                                                                                                                                                                                                                                                                                                                                                                               |
  | <code>Insert failed.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                  | <code>Issue : User Need Add Charge "Test Commisioning" In Two WO : PGI-WO-2412001092 & PGI-WO-2412001447 With detail : *Model : U-125PVY1H8 & U-100PVY1H8 *Work Type Testing & Commissioning, after Save is appear error : Review the errors on this page. First exception on row 0; first error: FIELD_CUSTOM_VALIDATION_EXCEPTION, The price book entry doesn t exist for this product: [PricebookEntryId Note : We check Price Book Already Mainain (For Detail Please see the attachment)</code> |
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
- `num_train_epochs`: 6
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 6
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: None
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `enable_jit_checkpoint`: False
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `use_cpu`: False
- `seed`: 42
- `data_seed`: None
- `bf16`: False
- `fp16`: True
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: -1
- `ddp_backend`: None
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
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
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `auto_find_batch_size`: False
- `full_determinism`: False
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `use_cache`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | Training Loss |
|:-----:|:----:|:-------------:|
| 1.0   | 500  | 1.2545        |
| 2.0   | 1000 | 0.5025        |
| 3.0   | 1500 | 0.2883        |
| 4.0   | 2000 | 0.1812        |
| 5.0   | 2500 | 0.1531        |
| 6.0   | 3000 | 0.1238        |


### Framework Versions
- Python: 3.11.14
- Sentence Transformers: 5.2.2
- Transformers: 5.1.0
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