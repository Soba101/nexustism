---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:5000
- loss:CombinedLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: 'Need to download Sales Org data (IAS2) Plants - IAL1, IAL2, IALA
    for the period of 010112 TO 260825 and have to give Japan (H.Q) on urgent basis.
    through MB51 we are unable to download the data as well as through SARE also.
    Pl help to provide data immediately. (Context: [sap | mm - material management]
    Group: capg l2 mm)'
  sentences:
  - 'User: Mafe FO 190004464 | DO# 551023651 User incorrectly input date as 16.01.2024
    in DO01 but input correct date in FO which is 16.01.2024, our billing and D/O
    was updated to correct date but system send 16.01.2024 date to customer. Pls kindly
    advise why system did not send the correct date to OSC. Pls note this is FTNS
    biz. (Context: [sap | sd - sales and distribution] [application/software | data
    - internal/external] Group: capg l2 sd brs)'
  - 'Already discussed this case with Mr. Sasitar. So strongly request to block such
    editing by user otherwise customer if purchase 2 items but paid/charge for only
    1 EW purchase then it is a loss to PM/NSC. NORHAYATI 3GWZ01562 NR-BX421WGWM DOP
    30/05/2022 W-910465 EXPIRY 29/05/2025 EW-00234925 - 30/5/2023 TO 29/05/2025 2GK400114
    W-910465 NR-BX471WGKM DOP 30/05/2022 EXPIRY 29/05/2025 (Context: [crm (d365, salesforce,
    genesis, pcube, hussmann services)] [application/software | program bug] Group:
    capg l2 crm)'
  - 'PRB0040413-To-Down Distribution KE28A test run job error Tcode: KE28A Variants:
    PM2501OBC1 Posting period : 12/2024 GID: 70P8140 ST22 error: Category Resou (Context:
    [sap] [application/software | job failure] Group: capg l2 sap basis)'
- source_sentence: 'ASC Transtronix PM-WO-2406000917. Request to change WO In-warranty
    and Claimable Flag to True since creation date 7/6/24 is before warranty expiry
    date 9/6/24. (Context: [crm (d365, salesforce, genesis, pcube, hussmann services)]
    [application/software | data - internal/external] Group: capg l2 crm)'
  sentences:
  - 'Asset no: 4100000002, Company code: 7100 created in PNP Scenario: Asset to be
    capitalized in Apr 2024, Depreciation for Apr 2024: to include Apr 23 - Mar 24
    + Apr 24 Cost of asset: USD 83,962.32 Monthly dep: USD 1,749.22 (Context: [sap
    | fico - finance & controlling] [application/software | data - internal/external]
    Group: piscap l2 fico)'
  - 'UNABLE TO TAKE INCOMING FOR FPL 0030315478 & 0030315472 (Context: [sap | mm -
    material management] [application/software | program bug] Group: capg l2 mm)'
  - 'NLP Process Flow & Documentation (Context: [crm (d365, salesforce, genesis, pcube,
    hussmann services)] [application/software | data - internal/external] Group: piscap
    l2 crm)'
- source_sentence: 'Invoice Number (Invoice/CN) In PGI-SO-2403008638 Is Not Appear
    After Generated Invoiced = (Invoice/CN) =0, it should be Appear, So help us to
    fix this (Context: [crm (d365, salesforce, genesis, pcube, hussmann services)]
    [application/software | user error] Group: capg l2 crm)'
  sentences:
  - 'Dear Basis team, in order to perform MRP run in TGP-900 we need to make sure
    how many users are currently active in PASDL, C1 company code. Please help to
    check and confirm. (Context: [sap | bc - basis] Group: piscap l2 sap basis)'
  - 'Dear Team Error occurred while processing the EDI transaction Interface Account
    Sync Subsidiary PPH API Name pana-crm-md-mgmt-papi Flow Direction inbound Source
    System PAPI End System Magento File Name NA Storage Path No Attachment Error Source
    Mulesoft Transaction ID b7179580-2fa9-11f0-bbbd-8ae32e30b1ed Error Summary 500
    HTTP:SERVICE_UNAVAILABLE Error Details ***********443/pph/rest/V1/customers/525387''
    failed: Timeout exceeded. Comments  Thanks and Regards, APAC Support, Panasonic
    (Context: [mulesoft/eai] [application/software | integration] Group: piscap l2
    mulesoft/soa)'
  - 'User: Sherlyn message": "Postal code 46200 must have the length 5 Please remove
    spacing before postal code and repost. If unclear, please contact me for verification.
    (Context: [eworkplace (eworkplace sharepoint & eworkplace servicenow)] [application/software
    | configuration] Group: capg l2 workflow (sn))'
- source_sentence: 'Dear Team Error occurred while processing the EDI transaction.
    Please find the details below and attached is the file associated to the transaction.
    Flow Direction inbound Subsidiary Source System Service Now End System BASIS File
    Name NA Storage Path NA API Name ework-snow-mgmt-eapi Error Source SAP Transaction
    ID 0a94d088-fa97-4ab7-bb49-5dc651133958 Error Summary 400 Unable to connect to
    SAP.Enter Correct User Name.. Please contact Basis at piscap-basis-team@sg.panasonic.com.
    Error Details Unable to connect to SAP.Enter Correct User Name.. Please contact
    Basis at piscap-basis-team@sg.panasonic.com. Comments  Thanks and Regards, APAC
    Support, Panasonic (Context: [mulesoft/eai] [application/software | user error]
    Group: piscap l2 mulesoft/soa)'
  sentences:
  - 'Attachment of Master Request does not appear in notification email receive by
    approver in PNP 900 and PNQ 900 PNP - Master Request W810099239 PNQ Test - Master
    Request W810074381 (Context: [sap | brs - budget rebate system] [application/software
    | user access] Group: piscap l2 sap basis)'
  - 'Z_ZUMMCP03_Q003 : Profitability segment fields are missing in Order Tracking
    (Context: [business insights (bi)] [application/software | data - internal/external]
    Group: capg l2 bw)'
  - '[PTP] DO outbound 9729043599 quantity is different between DO and idoc to send
    to Gotrans (Context: [sap | mm - material management] [application/software |
    job failure] Group: capg l2 mm)'
- source_sentence: 'Request to check if eTravel stage dropdown in eTravel header can
    be made as a read-only field. This is to reduce risk of users selecting Post in
    this dropdown instead of clicking Convert to Post (Context: [eworkplace (eworkplace
    sharepoint & eworkplace servicenow)] [application/software | configuration] Group:
    capg l2 workflow (sn))'
  sentences:
  - 'getting error in Y0GSD_0620 (Context: [sap | sd - sales and distribution] [application/software
    | data - internal/external] Group: capg l2 sd brs)'
  - 'Dear Team Error occurred while processing the EDI transaction Interface gid-emailid
    Subsidiary PAPAMY API Name pana-global-hriq-sapi Flow Direction inbound Source
    System End System File Name NA Storage Path No Attachment Error Source SuccessFactor-OData
    API Transaction ID 85275f10-5fd8-11f0-98ec-c29b10320a21 Error Summary 400 BAD_REQUEST
    Error Details Error response from SuccessFactors API Comments  Thanks and Regards,
    APAC Support, Panasonic (Context: [mulesoft/eai] [application/software | data
    - internal/external] Group: capg l2 mulesoft)'
  - 'User: Cindy When user try to find by vendor code: 29085 => hit error as attached.
    But when find via applicant name => no issue. (Context: [eworkplace (eworkplace
    sharepoint & eworkplace servicenow)] [application/software | configuration] Group:
    capg l2 workflow (sn))'
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
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'PeftModelForFeatureExtraction'})
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
    'Request to check if eTravel stage dropdown in eTravel header can be made as a read-only field. This is to reduce risk of users selecting Post in this dropdown instead of clicking Convert to Post (Context: [eworkplace (eworkplace sharepoint & eworkplace servicenow)] [application/software | configuration] Group: capg l2 workflow (sn))',
    'User: Cindy When user try to find by vendor code: 29085 => hit error as attached. But when find via applicant name => no issue. (Context: [eworkplace (eworkplace sharepoint & eworkplace servicenow)] [application/software | configuration] Group: capg l2 workflow (sn))',
    'Dear Team Error occurred while processing the EDI transaction Interface gid-emailid Subsidiary PAPAMY API Name pana-global-hriq-sapi Flow Direction inbound Source System End System File Name NA Storage Path No Attachment Error Source SuccessFactor-OData API Transaction ID 85275f10-5fd8-11f0-98ec-c29b10320a21 Error Summary 400 BAD_REQUEST Error Details Error response from SuccessFactors API Comments  Thanks and Regards, APAC Support, Panasonic (Context: [mulesoft/eai] [application/software | data - internal/external] Group: capg l2 mulesoft)',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9679, 0.4503],
#         [0.9679, 1.0000, 0.4034],
#         [0.4503, 0.4034, 1.0000]])
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
  | details | <ul><li>min: 32 tokens</li><li>mean: 113.69 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 28 tokens</li><li>mean: 113.21 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.47</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                          | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | label            |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>User Wing Ming . eSO-S1-24021099 posting fail . SAP SO unable to unlock for editing Pls see attach and resolve it with SAP as user need the SO to open up for editing to proceed her arrangement. (Context: [eworkplace (eworkplace sharepoint & eworkplace servicenow)] Group: capg l2 workflow (sn))</code> | <code>User: Daniela eCLM-S1-2500049 The taxable amount did not reflect correctly, should be SGD42.20. Pls help to check and advise. (Context: [eworkplace (eworkplace sharepoint & eworkplace servicenow)] [application/software \| configuration] Group: capg l2 workflow (sn))</code>                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>1.0</code> |
  | <code>See attachment for Request and Response json file, if available. Dealer Code : 5000016847 Date Time : 04/12/2024 09:24:38 Reason : SAP returned error (Context: [sap \| fico - finance & controlling] [application/software \| data - internal/external] Group: capg l2 fico)</code>                          | <code>Need help to retrigger the PIDSMY DOs because Mulesoft only received detail data, header data is 0 byte (empty) and no case mark data received too. See attached screenshot/files for reference. The batch contains 28 DOs. List of DO#s need to retrigger from MBP: 0255021671 0255021672 0255022154 0255022182 0255022183 0255022322 0255022323 0255022324 0255022325 0255022326 0255022332 0255022334 0255022335 0255022336 0255022337 0255022338 0255022339 0255022340 0255022341 0255022342 0255022343 0255022344 0255022345 0255022346 0255022347 0255022348 0255022349 0255022350 (Context: [sap \| sd - sales and distribution] [application/software \| integration] Group: capg l2 sd brs)</code> | <code>0.0</code> |
  | <code>GOS still showing after mark deletion. (Context: [sap \| sd - sales and distribution] Group: capg l2 abap)</code>                                                                                                                                                                                             | <code>1) No change idoc is triggered for Custom text field change in SMAP Indicator & CS Sales office in customer master, 2) No change idoc is triggered for Substitute partfield change (MARC - ZSUBP) in Material master We are making a change to capture this change so that data can be synced from SAP to SFDC successfully without any manual intervention. (Context: [sap] [application/software \| report] Group: capg l2 abap)</code>                                                                                                                                                                                                                                                                   | <code>1.0</code> |
* Loss: <code>__main__.CombinedLoss</code>

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
| 1.5974 | 500  | 1.7927        |
| 2.0    | 626  | -             |
| 2.9968 | 938  | -             |
| 3.0    | 939  | -             |
| 3.1949 | 1000 | 1.4220        |
| 4.0    | 1252 | -             |
| 1.0    | 313  | -             |
| 1.4984 | 469  | -             |
| 1.5974 | 500  | 1.2549        |
| 2.0    | 626  | -             |
| 2.9968 | 938  | -             |
| 3.0    | 939  | -             |
| 3.1949 | 1000 | 1.1915        |
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