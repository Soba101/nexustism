---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:5000
- loss:CosineSimilarityLoss
base_model: nomic-ai/nomic-embed-text-v1.5
widget:
- source_sentence: 'Finance user tried to post this eClaim49141 ,but appear this error
    message "There was some error while connecting to SAP. Contact Administrator.
    (Context: [eworkplace (eworkplace sharepoint & eworkplace servicenow)] [application/software
    | data - internal/external] Group: piscap l2 workflow)'
  sentences:
  - 'Hi Shirish/Khanh, can i get the cumulative balance for PPNDAP as of 30th (today)
    . need to check if there is any discrepancies before closing. (Context: [sap |
    fico - finance & controlling] [application/software | data - internal/external]
    Group: capg l2 fico)'
  - 'Unable to view approved document details in Sharepoint for eTravel and eClaim.
    Please check (Context: [eworkplace (eworkplace sharepoint & eworkplace servicenow)]
    [application/software | program bug] Group: piscap l2 workflow)'
  - 'User: Suck Keng D/O# 551023570 hit credit limit and user unable to perform cancellation
    because cargo is already packed but when trying to unpack, fields are grey off.
    We are unable to unpack. Pls kindly help to check & advise. Pls see attached the
    screenshots (Context: [sap | sd - sales and distribution] [application/software
    | program bug] Group: piscap l2 sd brs)'
- source_sentence: 'Dear Team Error occurred while processing the EDI transaction.
    Please find the details below and attached is the file associated to the transaction.
    Interface employee-minimaster-ridm Subsidiary PAPAMY API Name pana-global-hriq-sapi
    Flow Direction inbound Source System successFactors End System RIDM File Name
    ridm-empminimaster_fb8e2710-ddd3-11ef-8024-2a13ed2d59cd.json Storage Path /PAPAMY/ridm/employee-mini-master/ridm-empminimaster_fb8e2710-ddd3-11ef-8024-2a13ed2d59cd.json
    Error Source RIDM Transaction ID 13380b60-ddd4-11ef-8e24-e66798658974 Error Summary
    400 BAD_REQUEST Error Details Error response from RIDM API Comments  Thanks and
    Regards, APAC Support, Panasonic (Context: [mulesoft/eai] [application/software
    | user interface (ui)] Group: piscap l2 mulesoft/soa)'
  sentences:
  - 'eMaster - When Validate Material button is clicked, transaction keep running
    with no result since this morning (4th July 2025) (Context: [mulesoft/eai] [application/software
    | configuration] Group: capg l2 mulesoft)'
  - 'Please find the root cause why some message response eW can not receive from
    Genesis although the message was generated in SFDC side Here are two of them Request
    { "CUSTOMER_ADDRESS_FULL": "Hà N?i-C?u Gi?y-44 Tr?n Thái Tông", "IS_MARKETING":
    "1", "EW_NUMBER_ID": "", "CUSTOMER_PROVINCE_ID": "01", "STATUS": "1", "EW_START_DATE":
    "", "EW_STATUS": "", "CUSTOMER_DISTRICT_ID": "005", "WARRANTY_PROVINCE_ID": "01",
    "WARRANTY_CUSTOMER_ADDRESS": "Hà N?i-Tây H?-44 Ph??ng Th?y Khuê", "WARRANTY_DISTRICT_ID":
    "003", "MODEL_NAME": "CU-XPU18XKH-8B", "CUSTOMER_PHONE": "84904122118", "REGISTER_DATE":
    "2024-04-17", "EW_PERIOD": "", "CREATE_DATE": "2024-04-17", "CUSTOMER_NAME": "Hieu",
    "CUSTOMER_EMAIL": "", "PRODUCTION_DATE": "2024-04-17", "REGISTER_VIA": "ESV",
    "ENGINE_NO": "6788814544", "TAX_CODE": "", "WARRANTY_ID": "", "WARRANTY_EXPIRED_DATE":
    "2025-04-16" SFDC response : {"success":[{"ObjectName":"Customer","Details":["Customer
    Inserted/Updated. CUSTOMER_PHONE: 84904122118"]},{"ObjectName":"Asset","Details":["Asset
    Inserted/Updated. Engine_No: 6788814544"]},{"ObjectName":"Warranty Registration","Details":["Warranty
    Registration Inserted/Updated. WARRANTY_ID: W-8307990"]}],"exceptions":null,"error":null}
    request { "CUSTOMER_ADDRESS_FULL": "Hà N?i-C?u Gi?y-44 Tr?n Thái Tông", "IS_MARKETING":
    "1", "EW_NUMBER_ID": "", "CUSTOMER_PROVINCE_ID": "01", "STATUS": "1", "EW_START_DATE":
    "", "EW_STATUS": "", "CUSTOMER_DISTRICT_ID": "005", "WARRANTY_PROVINCE_ID": "01",
    "WARRANTY_CUSTOMER_ADDRESS": "Hà N?i-Tây H?-44 Ph??ng Th?y Khuê", "WARRANTY_DISTRICT_ID":
    "003", "MODEL_NAME": "CU-XPU18XKH-8", "CUSTOMER_PHONE": "84904122118", "REGISTER_DATE":
    "2024-04-17", "EW_PERIOD": "", "CREATE_DATE": "2024-04-17", "CUSTOMER_NAME": "Hieu",
    "CUSTOMER_EMAIL": "", "PRODUCTION_DATE": "2024-04-17", "REGISTER_VIA": "ESV",
    "ENGINE_NO": "6782225185", "TAX_CODE": "", "WARRANTY_ID": "", " SFDC response
    {"success":[{"ObjectName":"Customer","Details":["Customer Inserted/Updated. CUSTOMER_PHONE:
    84904122118"]},{"ObjectName":"Asset","Details":["Asset Inserted/Updated. Engine_No:
    6782225185"]},{"ObjectName":"Warranty Registration","Details":["Warranty Registration
    Inserted/Updated. WARRANTY_ID: W-8308062"]}],"exceptions":null,"error":null} (Context:
    [mulesoft/eai] [application/software | configuration] Group: capg l2 crm)'
  - 'account code for 23760001 in NZD to zerorise it (Context: [sap | fico - finance
    & controlling] Group: capg l2 fico)'
- source_sentence: 'GENESIS-SALESFORCE-PART STUCK AT PART REQUESTED STATUS - re-trigger
    SO to create picking slip and proceed to process. (Context: [crm (d365, salesforce,
    genesis, pcube, hussmann services)] [application/software | data - internal/external]
    Group: capg l2 crm)'
  sentences:
  - 'Dear Team Error occurred while processing the EDI transaction Interface Subsidiary
    No Subsidiary API Name pana-pagitp-mgmt-eapi Flow Direction Inbound Source System
    PAGITP End System SAP S4Hana File Name No File Name Storage Path No Attachment
    Error Source Mulesoft Transaction ID 4bdac8a2-987e-41b6-b7f6-02ff287f220e Error
    Summary 500 SOURCE_RESPONSE_SEND Error Details Client connection was closed Comments  Thanks
    and Regards, APAC Support, Panasonic (Context: [mulesoft/eai] [application/software
    | job failure] Group: piscap l2 mulesoft/soa)'
  - 'We raise 02 SO as below in SF but they were not sync to SAP => Could not create
    DO. PSV-SO-2404002965 (in PSV-WO-2404002569) and PSV-SO-2404002538 (in PSV-WO-2404001195)
    (Context: [crm (d365, salesforce, genesis, pcube, hussmann services)] [application/software
    | data - internal/external] Group: capg l2 crm)'
  - 'Hi Team, We have work order 00089740 , which is a maintenance work order. The
    first service appointment for a maintenance work order is meant to be scheduled
    for the duration of the starting hours mentioned on the Maintenance work rule
    (MWR) Appointments after the first one should default to an hour. (I have attached
    a snippet of this in Service Appointment Trigger Handler, I could be wrong ) I
    have noticed that the Duration is being set as 704 hours which is 29 days and
    that could be why this is blocking out the whole month SELECT Id, hm_Work_Order__r.WorkOrderNumber,hm_Duration_Calculated_On_Created__c,
    Status, SchedStartTime, SchedEndTime, Duration, DurationType, AppointmentNumber,
    CreatedDate FROM ServiceAppointment WHERE hm_Work_Order__r.WorkOrderNumber =''00089740''
    and hm_Work_Order__r.hm_Maintenance_Owner__r.name!=null order by CreatedDate desc
    Can we please check why this is happening Thanks Elaine (Context: [crm (d365,
    salesforce, genesis, pcube, hussmann services)] [application/software | configuration]
    Group: piscap l2 crm)'
- source_sentence: 'Dear Team Error occurred while processing the EDI transaction.
    Please find the details below and attached is the file associated to the transaction.
    Flow Direction inbound Subsidiary PAPVN-TL2 Source System INVOICE End System sap
    File Name NA Storage Path API Name sgst-fi-invoice-papi Error Source INVOICE Transaction
    ID 1701ef50-c654-11ef-a3ad-a699d2f48aa7 Error Summary FPT thông báo: resData.map
    is not a function Error Details FPT thông báo: resData.map is not a function Comments
    Unable to retrive file from Backup location  Thanks and Regards, APAC Support,
    Panasonic (Context: [mulesoft/eai] [application/software | data - internal/external]
    Group: piscap l2 mulesoft/soa)'
  sentences:
  - 'CPC Part Center Team, Received Material No 1610Z0801AM-S1, qty of 100 pcs in
    (PO : 7973002952 / DO : 8100001468) And failed to receive into system due to blank
    delivery quantity remain ( 100 pcs ) as below screen capture TCode VL31N Details
    on attached On Servicenow (Context: [sap | wm - warehouse management] [application/software
    | data - internal/external] Group: capg l2 wm)'
  - 'Part number DMUD4CXDK is non-GCS part, maintained in SAP with product hierarchy
    XXXXXXXXXXM10A504J but no DCAT value shown in BI Z_ZUGLCP01_Q003 (sales results
    DCM PIDS). (Context: [business insights (bi)] [application/software | data - internal/external]
    Group: capg l2 bw)'
  - 'Pre-Travel for MD has incorrect number of days in daily allowance. Below is the
    pre-travel details:- Travel Period: 15th March - 23rd March 15th March: 0.5days
    + 1 additional day (red-eye flight) 16th - 23rd March: 8 days Should be 9.5days
    but application is displaying 8.5days only (Context: [eworkplace (eworkplace sharepoint
    & eworkplace servicenow)] [application/software | configuration] Group: capg l2
    workflow (sn))'
- source_sentence: 'We are humbly requesting to analyze root cause of the sudden buildup
    of Production Order Variance during order settlement. Plant: 34P3 Period: January
    2025 Note: Please refer to attached screenshot of one Prod. Order that has big
    variance. (Context: [sap | fico - finance & controlling] [application/software
    | configuration] Group: piscap l2 fico)'
  sentences:
  - 'User : Mike During GST audit at customer site, accounting notice one invoice
    print out , there is 1 cent different from the system after converted from USD
    to SGD . ECC6 Invoice billing # : 581196698 Print out : 7091.41 SGD System : 7091.41
    pls kindly check and confirm that above issue will not happen in Hana system .
    Pls replica the above transaction and confirm by return . (Context: [sap | sd
    - sales and distribution] [application/software | program bug] Group: capg l2
    sd brs)'
  - 'delivery order does not show reprint if printed for second time. (Context: [sap
    | wm - warehouse management] [application/software | report] Group: capg l2 sd
    brs)'
  - 'ZPM and KE30: -2,796,260.52 1VK: -2,780,660.52 Difference: 15,600. 15.600 of
    70010007 / 70010002 did not go to 1VK S70010002, but the amount go to Ke30 S70010002.
    (Context: [sap | fico - finance & controlling] [application/software | report]
    Group: piscap l2 fico)'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on nomic-ai/nomic-embed-text-v1.5

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) <!-- at revision e5cf08aadaa33385f5990def41f7a23405aec398 -->
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
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'PeftModelForFeatureExtraction'})
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
    'We are humbly requesting to analyze root cause of the sudden buildup of Production Order Variance during order settlement. Plant: 34P3 Period: January 2025 Note: Please refer to attached screenshot of one Prod. Order that has big variance. (Context: [sap | fico - finance & controlling] [application/software | configuration] Group: piscap l2 fico)',
    'ZPM and KE30: -2,796,260.52 1VK: -2,780,660.52 Difference: 15,600. 15.600 of 70010007 / 70010002 did not go to 1VK S70010002, but the amount go to Ke30 S70010002. (Context: [sap | fico - finance & controlling] [application/software | report] Group: piscap l2 fico)',
    'delivery order does not show reprint if printed for second time. (Context: [sap | wm - warehouse management] [application/software | report] Group: capg l2 sd brs)',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9799, 0.0248],
#         [0.9799, 1.0000, 0.0156],
#         [0.0248, 0.0156, 1.0000]])
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

* Size: 5,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                           | sentence_1                                                                           | label                                                          |
  |:--------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                               | string                                                                               | float                                                          |
  | details | <ul><li>min: 33 tokens</li><li>mean: 113.21 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 22 tokens</li><li>mean: 113.57 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.47</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Dear Team Error occurred while processing the EDI transaction. Please find the details below and attached is the file associated to the transaction. Interface GITP Subsidiary PIDSAP API Name inds-global-if-mgmt-papi Flow Direction Outbound Source System SAP End System IBMMQ File Name 0000000001917892_91768a80-c35b-11ef-935d-6ed695ca58b3.xml Storage Path /INDS/prod/outbound/nocompany/ordrsp/0000000001917892_91768a80-c35b-11ef-935d-6ed695ca58b3.xml Error Source Mulesoft Transaction ID 91768a80-c35b-11ef-935d-6ed695ca58b3 Error Summary 500 CONNECTIVITY Error Details ***********443/api/v1/aleaud' failed: Remotely closed. Comments Unable to find 0000000001917892_91768a80-c35b-11ef-935d-6ed695ca58b3.xml from Backup location  Thanks and Regards, APAC Support, Panasonic (Context: [mulesoft/eai] [application/software \| integration] Group: piscap l2 mulesoft/soa)</code> | <code>Interface Subsidiary PIDSAP API Name pana-marketing-mgmt-papi Flow Direction inbound Source System PAPI End System File Name NA Storage Path No Attachment Error Source Mulesoft Transaction ID fc830d10-fc34-11ee-bb03-564c6b14c4ed, 6223a200-fc36-11ee-bb03-564c6b14c4ed Error Summary 500 SERVICE_UNAVAILABLE Error Details ***********443/api/v1/ecommerce/products' failed: service unavailable (503). ***********443/api/v1/ecommerce/product-locales' failed: service unavailable (503). (Context: [mulesoft/eai] [application/software \| integration] Group: piscap l2 mulesoft/soa)</code> | <code>1.0</code> |
  | <code>Dear Team Error occurred while processing the EDI transaction Interface Subsidiary PIDSAP API Name ext-partners-order-mgmt-papi Flow Direction Inbound Source System SAP End System NA File Name NA Storage Path No Attachment Error Source Mulesoft Transaction ID 59b9af70-7f83-11f0-8f00-d2613a267bb8 Error Summary 500 UNKNOWN Error Details Exception was found writing to file '/INDS/inbound/pidsmy/cpo/avnet/Avnet AS2 Summary Report.xls' Comments  Thanks and Regards, APAC Support, Panasonic (Context: [mulesoft/eai] [application/software \| job failure] Group: piscap l2 mulesoft/soa)</code>                                                                                                                                                                                                                                                                                             | <code>Dear Team Error occurred while processing the EDI transaction Interface gid-emailid Subsidiary PAPAMY API Name pana-global-hriq-sapi Flow Direction inbound Source System End System File Name NA Storage Path No Attachment Error Source SuccessFactor-OData API Transaction ID 85275f10-5fd8-11f0-98ec-c29b10320a21 Error Summary 400 BAD_REQUEST Error Details Error response from SuccessFactors API Comments  Thanks and Regards, APAC Support, Panasonic (Context: [mulesoft/eai] [application/software \| data - internal/external] Group: capg l2 mulesoft)</code>                           | <code>0.0</code> |
  | <code>approver didn't receive email notification Deal Request# S610054064 for Approval (Context: [sap \| bc - basis] [application/software \| user error] Group: capg l2 sap basis)</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | <code>Please review the Roles assigned to PMPC. Some of Role's Authorization Object have a wildcard " * " value. As background, PMPC Ordering member was able to create Purchase Order under unauthorized Plant and other Companies. Please investigate why SAP ID P70D2996 has authorization to create under Plant 34P1, 34P3, 11P1 and 11P2. Note: Please see attached roles extracted from table AGR_1251 (Context: [sap \| bc - basis] [application/software \| user access] Group: capg l2 sap basis)</code>                                                                                          | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 4
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `do_predict`: False
- `eval_strategy`: steps
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
- `num_train_epochs`: 4
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
- `fp16`: False
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
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 1.0    | 313  | -             |
| 1.4984 | 469  | -             |
| 1.5974 | 500  | 0.1072        |
| 2.0    | 626  | -             |
| 2.9968 | 938  | -             |
| 3.0    | 939  | -             |
| 3.1949 | 1000 | 0.0246        |
| 4.0    | 1252 | -             |
| 1.0    | 313  | -             |
| 1.4984 | 469  | -             |
| 1.5974 | 500  | 0.0181        |
| 2.0    | 626  | -             |
| 2.9968 | 938  | -             |
| 3.0    | 939  | -             |
| 3.1949 | 1000 | 0.0125        |
| 4.0    | 1252 | -             |
| 1.0    | 313  | -             |


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