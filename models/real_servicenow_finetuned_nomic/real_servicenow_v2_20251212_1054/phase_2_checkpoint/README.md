---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:1278
- loss:MultipleNegativesRankingLoss
- dataset_size:1269
base_model: nomic-ai/nomic-embed-text-v1.5
widget:
- source_sentence: 'as check, we already invoice the Sales Order but when we download
    the attachment, Invoice No : and Invoice Date : was empty. kindly seek your help
    to check. refer to attachment below. thanks. (Context: [crm (d365, salesforce,
    genesis, pcube, hussmann services)] [application/software | data - internal/external]
    Group: capg l2 crm)'
  sentences:
  - 'PSV-WO-2507006235 PSV-SO-2507002499 (Context: [crm (d365, salesforce, genesis,
    pcube, hussmann services)] [application/software | integration] Group: capg l2
    crm)'
  - 'UNABLE TO ACCESS THE SAP SYSTEM DUE TO USERNAME LOCKED. (Context: [sap | bc -
    basis] [application/software | user access] Group: capg l2 sap basis)'
  - 'as check with tech, part already received and set done repair. kindly seek your
    help to change part from Dlv Processing to Delivered. refer to attachment below.
    thanks. (Context: [crm (d365, salesforce, genesis, pcube, hussmann services)]
    [application/software | integration] Group: capg l2 crm)'
- source_sentence: 'Dear Team Error occurred while processing the EDI transaction
    Interface Subsidiary No Subsidiary API Name pana-pagitp-mgmt-eapi Flow Direction
    Inbound Source System PAGITP End System SAP S4Hana File Name No File Name Storage
    Path No Attachment Error Source Mulesoft Transaction ID 150d9640-db85-11ef-b84b-c20fc72a105a
    Error Summary 500 ILLEGAL_PATH Error Details Path ''/seglink/PIDSKR/prod/fromSEGLink/OB''
    doesn''t exist Comments  Thanks and Regards, APAC Support, Panasonic (Context:
    [mulesoft/eai] [application/software | integration] Group: piscap l2 mulesoft/soa)'
  sentences:
  - 'Dear Team Error occurred while processing the EDI transaction Interface Subsidiary
    No Subsidiary API Name pana-pagitp-mgmt-eapi Flow Direction Inbound Source System
    PAGITP End System SAP S4Hana File Name No File Name Storage Path No Attachment
    Error Source Mulesoft Transaction ID 89816ec0-b9f2-11ef-b830-0a52343ebe1e Error
    Summary 500 CONNECTION_TIMEOUT Error Details Could not establish SFTP connection
    with host: ''10.86.48.62'' at port: ''22'' - timeout: socket is not established
    Comments  Thanks and Regards, APAC Support, Panasonic (Context: [mulesoft/eai]
    [application/software | job failure] Group: piscap l2 mulesoft/soa)'
  - 'Dear Team Error occurred while processing the EDI transaction Interface Subsidiary
    No Subsidiary API Name pana-pagitp-mgmt-eapi Flow Direction Inbound Source System
    PAGITP End System SAP S4Hana File Name No File Name Storage Path No Attachment
    Error Source Mulesoft Transaction ID 3125178f-8fb7-4b85-b30e-aa8fd0090991 Error
    Summary 500 SOURCE_RESPONSE_SEND Error Details Client connection was closed Comments  Thanks
    and Regards, APAC Support, Panasonic (Context: [mulesoft/eai] [application/software
    | integration] Group: piscap l2 mulesoft/soa)'
  - 'Dear Team Error occurred while processing the EDI transaction Interface PANA.CART-PRD.MARKETING.Q
    Subsidiary PSV API Name pana-sf-mc-sapi Flow Direction inbound Source System PAPI
    End System Salesforce Marketing Cloud File Name NA Storage Path No Attachment
    Error Source Mulesoft Transaction ID 57486e20-072e-11f0-a511-6a58f71b67be Error
    Summary 500 COMPOSITE_ROUTING Error Details COMPOSITE_ROUTING: Exception(s) were
    found for route(s): Route 0: org.mule.runtime.core.api.retry.policy.RetryPolicyExhaustedException:
    ''until-successful'' retries exhausted Comments  Thanks and Regards, APAC Support,
    Panasonic (Context: [mulesoft/eai] [application/software | integration] Group:
    piscap l2 mulesoft/soa)'
- source_sentence: 'No daily allowance rate maintained for United Kingdom (Business
    Trip). When I checked Go-Live masters, we have a row dedicated to United Kingdom
    from Sharepoint. (Context: [eworkplace (eworkplace sharepoint & eworkplace servicenow)]
    [application/software | configuration] Group: piscap l2 workflow (sn))'
  sentences:
  - 'Dear Team Error occurred while processing the EDI transaction. Please find the
    details below and attached is the file associated to the transaction. Interface
    GITP Subsidiary PIDSAP API Name inds-global-if-mgmt-papi Flow Direction Outbound
    Source System SAP End System IBMMQ File Name 0000000001727283_0684bc60-ad3c-11ef-bd0c-96a5d4410501.xml
    Storage Path /INDS/prod/outbound/nocompany/ordrsp/0000000001727283_0684bc60-ad3c-11ef-bd0c-96a5d4410501.xml
    Error Source Mulesoft Transaction ID 0684bc60-ad3c-11ef-bd0c-96a5d4410501 Error
    Summary 500 CONNECTIVITY Error Details ***********443/api/v1/sd/order/confirmation''
    failed: Remotely closed. Comments Unable to find 0000000001727283_0684bc60-ad3c-11ef-bd0c-96a5d4410501.xml
    from Backup location  Thanks and Regards, APAC Support, Panasonic (Context: [mulesoft/eai]
    [application/software | integration] Group: piscap l2 mulesoft/soa)'
  - 'PART SHOWN DISCONTINUED. PART STILL HAVE 1 STOCK TO ORDER. PART NUMBER : ARBGLA100051
    PART DESCRIPTION : LED PCB AS. BL9 (Context: [crm (d365, salesforce, genesis,
    pcube, hussmann services)] [application/software | database] Group: capg l2 crm)'
  - 'Jiro Nakami GID : 70Q9374 (Context: [eworkplace (eworkplace sharepoint & eworkplace
    servicenow)] [application/software | user interface (ui)] Group: piscap l2 workflow)'
- source_sentence: 'Dear Team Error occurred while processing the EDI transaction
    Interface Subsidiary No Subsidiary API Name pana-pagitp-mgmt-eapi Flow Direction
    Inbound Source System PAGITP End System SAP S4Hana File Name No File Name Storage
    Path No Attachment Error Source Mulesoft Transaction ID 6034f920-01b8-11f0-97d8-8624babe0d48
    Error Summary 500 CONNECTION_TIMEOUT Error Details Could not establish SFTP connection
    with host: ''10.86.48.62'' at port: ''22'' - timeout: socket is not established
    Comments  Thanks and Regards, APAC Support, Panasonic (Context: [mulesoft/eai]
    [application/software | job failure] Group: piscap l2 mulesoft/soa)'
  sentences:
  - 'Dear Team Error occurred while processing the EDI transaction. Please find the
    details below and attached is the file associated to the transaction. Interface
    employee-minimaster-ridm Subsidiary PAPAMY API Name pana-global-hriq-sapi Flow
    Direction inbound Source System successFactors End System RIDM File Name ridm-empminimaster_09406a60-b365-11ef-86c5-aa5e283632a7.json
    Storage Path /PAPAMY/ridm/employee-mini-master/ridm-empminimaster_09406a60-b365-11ef-86c5-aa5e283632a7.json
    Error Source RIDM Transaction ID 1cc16581-b365-11ef-b8fd-7210620ee764 Error Summary
    400 BAD_REQUEST Error Details Error response from RIDM API Comments  Thanks and
    Regards, APAC Support, Panasonic (Context: [mulesoft/eai] [application/software
    | data - internal/external] Group: piscap l2 mulesoft/soa)'
  - 'Dear Team Error occurred while processing the EDI transaction Interface Subsidiary
    No Subsidiary API Name pana-pagitp-mgmt-eapi Flow Direction Inbound Source System
    PAGITP End System SAP S4Hana File Name No File Name Storage Path No Attachment
    Error Source Mulesoft Transaction ID 5b0f3224-50f8-48a0-b9c0-a5099812d57b Error
    Summary 500 SOURCE_RESPONSE_SEND Error Details Client connection was closed Comments  Thanks
    and Regards, APAC Support, Panasonic (Context: [mulesoft/eai] [network | access
    point] Group: piscap l2 mulesoft/soa)'
  - 'Dear Team Error occurred while processing the EDI transaction Interface avnet
    Subsidiary PIDSMY API Name ext-partners-order-mgmt-papi Flow Direction Inbound
    Source System SAP End System NA File Name NA Storage Path No Attachment Error
    Source Mulesoft Transaction ID 935ff643-a8c8-11ef-8e21-460d2b807079 Error Summary
    500 TIMEOUT Error Details ***********443/api/v1/file-reference-data/YELM214X''
    failed: Timeout exceeded. Comments  Thanks and Regards, APAC Support, Panasonic
    (Context: [mulesoft/eai] [application/software | job failure] Group: piscap l2
    mulesoft/soa)'
- source_sentence: 'Dear Team Error occurred while processing the EDI transaction
    Interface Subsidiary No Subsidiary API Name pana-pagitp-mgmt-eapi Flow Direction
    Inbound Source System PAGITP End System SAP S4Hana File Name No File Name Storage
    Path No Attachment Error Source Mulesoft Transaction ID 7e8bf560-2d3a-11f0-ae4d-fa5087e70212
    Error Summary 500 CONNECTIVITY Error Details Could not establish SFTP connection
    with host: ''10.86.48.62'' at port: ''22'' - Session.connect: java.net.SocketTimeoutException:
    Read timed out Comments  Thanks and Regards, APAC Support, Panasonic (Context:
    [mulesoft/eai] [application/software | integration] Group: piscap l2 mulesoft/soa)'
  sentences:
  - 'Dear Team Error occurred while processing the EDI transaction. Please find the
    details below and attached is the file associated to the transaction. Interface
    employee-minimaster-ridm Subsidiary PAPAMY API Name pana-global-hriq-sapi Flow
    Direction inbound Source System successFactors End System RIDM File Name ridm-empminimaster_83bc2710-48b2-11f0-b153-222e0650bd16.json
    Storage Path /PAPAMY/ridm/employee-mini-master/ridm-empminimaster_83bc2710-48b2-11f0-b153-222e0650bd16.json
    Error Source RIDM Transaction ID 86295680-48b2-11f0-aa32-1af8b3654c26 Error Summary
    400 BAD_REQUEST Error Details Error response from RIDM API Comments  Thanks and
    Regards, APAC Support, Panasonic (Context: [mulesoft/eai] [application/software
    | data - internal/external] Group: piscap l2 mulesoft/soa)'
  - 'Dear Team Error occurred while processing the EDI transaction Interface Subsidiary
    No Subsidiary API Name pana-pagitp-mgmt-eapi Flow Direction Inbound Source System
    PAGITP End System SAP S4Hana File Name No File Name Storage Path No Attachment
    Error Source Mulesoft Transaction ID 529c8e50-3128-11f0-ae4d-fa5087e70212 Error
    Summary 500 CONNECTIVITY Error Details Could not establish SFTP connection with
    host: ''10.86.48.62'' at port: ''22'' - Session.connect: java.net.SocketTimeoutException:
    Read timed out Comments  Thanks and Regards, APAC Support, Panasonic (Context:
    [mulesoft/eai] [application/software | integration] Group: piscap l2 mulesoft/soa)'
  - 'Dear Team Error occurred while processing the EDI transaction Interface GITP
    Subsidiary NA API Name inds-global-if-mgmt-papi Flow Direction Outbound Source
    System SAP End System IBMMQ File Name NA Storage Path No Attachment Error Source
    Mulesoft Transaction ID 8dccced0-151d-11f0-8ae8-7ea2b6d5ca67 Error Summary 501
    EXPRESSION Error Details Expression Error Occured Comments  Thanks and Regards,
    APAC Support, Panasonic (Context: [mulesoft/eai] [application/software | job failure]
    Group: piscap l2 mulesoft/soa)'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on nomic-ai/nomic-embed-text-v1.5

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) <!-- at revision e5cf08aadaa33385f5990def41f7a23405aec398 -->
- **Maximum Sequence Length:** 512 tokens
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
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'NomicBertModel'})
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
    "Dear Team Error occurred while processing the EDI transaction Interface Subsidiary No Subsidiary API Name pana-pagitp-mgmt-eapi Flow Direction Inbound Source System PAGITP End System SAP S4Hana File Name No File Name Storage Path No Attachment Error Source Mulesoft Transaction ID 7e8bf560-2d3a-11f0-ae4d-fa5087e70212 Error Summary 500 CONNECTIVITY Error Details Could not establish SFTP connection with host: '10.86.48.62' at port: '22' - Session.connect: java.net.SocketTimeoutException: Read timed out Comments  Thanks and Regards, APAC Support, Panasonic (Context: [mulesoft/eai] [application/software | integration] Group: piscap l2 mulesoft/soa)",
    "Dear Team Error occurred while processing the EDI transaction Interface Subsidiary No Subsidiary API Name pana-pagitp-mgmt-eapi Flow Direction Inbound Source System PAGITP End System SAP S4Hana File Name No File Name Storage Path No Attachment Error Source Mulesoft Transaction ID 529c8e50-3128-11f0-ae4d-fa5087e70212 Error Summary 500 CONNECTIVITY Error Details Could not establish SFTP connection with host: '10.86.48.62' at port: '22' - Session.connect: java.net.SocketTimeoutException: Read timed out Comments  Thanks and Regards, APAC Support, Panasonic (Context: [mulesoft/eai] [application/software | integration] Group: piscap l2 mulesoft/soa)",
    'Dear Team Error occurred while processing the EDI transaction. Please find the details below and attached is the file associated to the transaction. Interface employee-minimaster-ridm Subsidiary PAPAMY API Name pana-global-hriq-sapi Flow Direction inbound Source System successFactors End System RIDM File Name ridm-empminimaster_83bc2710-48b2-11f0-b153-222e0650bd16.json Storage Path /PAPAMY/ridm/employee-mini-master/ridm-empminimaster_83bc2710-48b2-11f0-b153-222e0650bd16.json Error Source RIDM Transaction ID 86295680-48b2-11f0-aa32-1af8b3654c26 Error Summary 400 BAD_REQUEST Error Details Error response from RIDM API Comments  Thanks and Regards, APAC Support, Panasonic (Context: [mulesoft/eai] [application/software | data - internal/external] Group: piscap l2 mulesoft/soa)',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9995, 0.0272],
#         [0.9995, 1.0000, 0.0274],
#         [0.0272, 0.0274, 1.0000]])
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

* Size: 1,269 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                           | sentence_1                                                                          | label                                                         |
  |:--------|:-------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                               | string                                                                              | float                                                         |
  | details | <ul><li>min: 34 tokens</li><li>mean: 157.06 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 33 tokens</li><li>mean: 156.1 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 1.0</li><li>mean: 1.0</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | label            |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>These ebill stay here even though already approved (Context: [eworkplace (eworkplace sharepoint & eworkplace servicenow)] [application/software \| configuration] Group: capg l2 workflow (sn))</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | <code>User: Sofia Loraine Issue: eSO-S1-24005475 Posting Failed Message": "Sales Order 110038198 is not yet send to eWork", (Context: [eworkplace (eworkplace sharepoint & eworkplace servicenow)] [application/software \| integration] Group: piscap l2 workflow (sn))</code>                                                                                                                                                                                                                                                                                                                                | <code>1.0</code> |
  | <code>can't select non bp type/category in eIAF19876 (Context: [eworkplace (eworkplace sharepoint & eworkplace servicenow)] [application/software \| user interface (ui)] Group: piscap l2 workflow)</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | <code>Submit button missing from all ServiceNow applications in production environment (Context: [eworkplace (eworkplace sharepoint & eworkplace servicenow)] [application/software \| user interface (ui)] Group: capg l2 workflow (sn))</code>                                                                                                                                                                                                                                                                                                                                                               | <code>1.0</code> |
  | <code>Dear Team Error occurred while processing the EDI transaction Interface avnet Subsidiary PIDSMY API Name ext-partners-order-mgmt-papi Flow Direction Inbound Source System SAP End System NA File Name NA Storage Path No Attachment Error Source Mulesoft Transaction ID 6f641750-ffec-11ef-a92a-96c9562cc24c Error Summary 500 UNKNOWN Error Details org.mule.runtime.api.exception.MuleRuntimeException: Exception was found trying to retrieve the contents of file /INDS/inbound/pidsmy/cpo/avnet/Avnet AS2 Summary Report.xls Comments  Thanks and Regards, APAC Support, Panasonic (Context: [mulesoft/eai] [application/software \| job failure] Group: piscap l2 mulesoft/soa)</code> | <code>Dear Team Error occurred while processing the EDI transaction Interface avnet Subsidiary PIDSMY API Name ext-partners-order-mgmt-papi Flow Direction Inbound Source System SAP End System NA File Name NA Storage Path No Attachment Error Source Mulesoft Transaction ID ed183590-d18f-11ef-bd5b-f2af15d561b6 Error Summary 501 RETRY_EXHAUSTED Error Details ***********443/api/v1/PIDSAP/customer-material-table' failed: Remotely closed. Comments  Thanks and Regards, APAC Support, Panasonic (Context: [mulesoft/eai] [application/software \| integration] Group: piscap l2 mulesoft/soa)</code> | <code>1.0</code> |
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
- `num_train_epochs`: 2
- `fp16`: True
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
- `num_train_epochs`: 2
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
- `optim`: adamw_torch
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
| Epoch | Step |
|:-----:|:----:|
| 1.0   | 80   |
| 2.0   | 160  |
| 1.0   | 80   |
| 2.0   | 160  |


### Framework Versions
- Python: 3.11.14
- Sentence Transformers: 5.1.2
- Transformers: 4.57.3
- PyTorch: 2.5.1+cu121
- Accelerate: 1.12.0
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