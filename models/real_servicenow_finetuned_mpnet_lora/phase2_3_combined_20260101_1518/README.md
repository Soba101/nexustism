---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:10000
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: 'INC0055364 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Interface       Biometric - Time Clock Data

    Subsidiary      PAPAMY

    API Name        pana-hriq-mgmt-papi

    Flow Direction  inbound

    Source System   SAP ECC FI - FTP

    End System      pana-global-hriq-sapi

    File Name       /HR/BioTime/PRD/ERROR/ERROR_Panasonic 202504061030_all_96b5cfa0-1442-11f0-9474-4aee0e81b5b8_3.txt

    Storage Path    /PAPAMY/biometric/time-clock-data/Panasonic 202504061030_all_96b5cfa0-1442-11f0-9474-4aee0e81b5b8.txt

    Error Source    Mulesoft

    Transaction ID  96b5cfa0-1442-11f0-9474-4aee0e81b5b8

    Error Summary   501 RUNTIME_ERROR

    Error Details   Invalid payload

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic duplicated issue'
  sentences:
  - 'INC0057171 Dear Team


    Error occurred while processing the EDI transaction


    Interface       NA

    Subsidiary      NA

    API Name        pana-sdesk-ext-eapi

    Flow Direction  outbound

    Source System   NA

    End System      NA

    File Name       No File Name

    Storage Path    No Attachment

    Error Source    pana-sdesk-ext-eapi

    Transaction ID  eddcd363-7ac8-40bd-9731-1a25c5923cca

    Error Summary   500 EXPRESSION

    Error Details   "EXPRESSION: \"Unexpected character ''\u0009'' at payload@[1:533]
    (line:column), expected ''\"'', while reading `payload` as Json.\n \n1| {\"sitRefNo\":\"ISR-10219\",\"fromWhse\":\"A1\",\"reasonWode\":\"PM1A\",\"custOrderNo\":\"26-013\",\"shipToCompanyName\":\"MoniqueMckenzie\",\"shipToContactName\":\"MoniqueMckenzie\",\"shipToContactPhone\":\"0423879749\",\"shipToAddress1\":\"31
    Anderson St\",\"shipToAddress2\":\"\",\"shipToAddress3\":\"SCARBOROUGH\",\"shipToAddress4\":\"QLD\",\"shipToPostCode\":\"4020\",\"routNo\":\"HOM\",\"approvalStage\":\"Approved
    by Warehouse Operations\",\"transferStatus\":\"\",\"partnerOrderNo\":\"\",\"ProcessStatus\":\"N\",\"stockPartDetails\":[{\"sitRefNo\":\"ISR-10219\",\"itemLineNo\":1,\"itemNo\":\"ES-WR51-P541\t\",\"orderQty\":1,\"glNo\":\"91105942\"}]}\n
    ^\" evaluating expression: \"%dw 2.0\noutput application/json\n---\n{\n\tcompany:
    attributes.uriParams.''company'',\n\tinterface: Mule::p(''api.interface.stock-parts''),\n\tinterfaceType:
    ''REST'',\n\tsource: attributes.uriParams.''company'',\n\ttarget: Mule::p(''api.target.mssql''),\n\taction:
    Mule::p(''api.action.post'') ++ \"-\" ++ Mule::p(''api.subAction.data-received''),\n\tstatus:
    Mule::p(''api.status.initiate''),\n\tpayload: payload,\n\tpayloadType: Mule::p(''api.payload.type.request''),\n\ttracePoint:
    \"BEFORE_REQUEST\",\n\tpriority: \"INFO\",\n\terrorSource: \"\", \n errorCode:
    \"\", \n errorMessage: \"\", \n errorPayload: \"\"\n}\"."

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Encoding issue on zoho'
  - "INC0027815 Service Category already assigned as Plasma/LED display panel/video\
    \ wall 40 - 54 inches. Hi Pang,\r\n\r\nKindly Inform you that PM-WO-2403004551\
    \ charge group has been updated to Plasma/LED display panel/video wall 40 - 54\
    \ inches ."
  - "INC0047228 User: Suck Keng\r\nD/O# 551023570 hit credit limit and user unable\
    \ to perform cancellation because cargo is already packed but when trying to unpack,\
    \ fields are grey off. We are unable to unpack. Pls kindly help to check & advise.\
    \ \r\n\r\nPls see attached the screenshots Created fix ticket CHG0032359 to modify\
    \ the program so that user can unpack & delete the DO"
- source_sentence: "INC0063810 Hi Team \r\nThe below idocs aew successful but have\
    \ not reached sfdc \r\nThe purchase orders are still stuck in SAP \r\nCan you\
    \ please repush the below idocs\r\n\r\n\r\nIdoc 5390508\r\nIdoc 5383266\r\n\r\n\
    Thanks \r\nElaine Data Issue - missing PO lines"
  sentences:
  - 'INC0069328 Dear Team


    Error occurred while processing the EDI transaction


    Interface       GITP

    Subsidiary      PIDSAP

    API Name        inds-global-if-mgmt-papi

    Flow Direction  Inbound

    Source System   IBMMQ

    End System      SAP

    File Name       NA

    Storage Path    No Attachment

    Error Source    Mulesoft

    Transaction ID  ID:363134323538393234303030303030303030303020202020

    Error Summary   502 BAD_GATEWAY

    Error Details   ***********443/api/v1/sender-receiver-code-reference-data'' failed:
    bad gateway (502).

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic'
  - INC0034662 Factory transmitted FPL SSL4H2621382 with incorrect line item but correct
    part number. However, system posted invoice based on part numbers from line item,
    not part number sent by factory. This occured because factory PIC incorrectly
    entered wrong line item number but please check and confirm what is our system
    logic to post billing based on FPL data. Is it by line item number or part number?
    Provided details to the user. User confirmed to close the ticket.
  - "INC0042870 User: Geri\r\n\r\nmessage\": \"Partner Function: Plausibility check\
    \ failed\"\r\nmessage\": \"Partner Function: Is a required entry field\"\r\nmessage\"\
    : \"Customer 80950657 / 7030 / 11 / 11: Data is incomplete; check data\"\r\n\r\
    \nPlease remove partner function item number 000006 and repost. Closed by Caller"
- source_sentence: "INC0026641 Pls help me to check setup of CS Dien Lanh Quoc account\
    \ - 5000027163\r\nAll WOs don't update claim fee after change WO status to \"\
    Work completed\"\r\nEX: PSV-WO-2407003561\r\nPSV-WO-2407001479 Issue:error claim\
    \ fee in WO of CS Dien Lanh Quoc\r\nAction steps: I have updated the owner name\
    \ of the Asc CS Dien Lanh Quoc account - 5000027163  to Hoang Kim Nguyen. No actions\
    \ to be required from sfdc side. Hence closing the incident."
  sentences:
  - "INC0028698 Material TH-55NX600G-RS is already having its PIR (purchase price)\
    \ maintained in SAP, but still does not shown in SFDC.\r\nWe already tried to\
    \ re-synchronized to SFDC, but not succesful\r\nKindly Please help to fix this\
    \ issue, Thanks Closed by Caller"
  - INC0051368 Pls resend EWS PSV-EWS-2502000230/ PSV-EWS-2502000233/ PSV-EWS-2502000234/
    PSV-EWS-2502000235 with posting date is the date you resend. run  script  se-send
    order to SAP
  - "INC0054489 Error in Section: CREATE_INB_DELIVERY occurred \r\nError in Section:\
    \ POST_DELIVERY occurred\r\nPO 400036974 item 00050 - error processing delivery\
    \ \r\nInvoice IPL(8100010253 )/SAP(5100213312 ) from 20.03.2025 already exist\r\
    \nError!! Invoice already exists for IPS Invoice Number (8100010253 )\r\n\r\n\
    INV. 8100010253 Factory sent incorrect packing data due to this system failed\
    \ to create inbound delivery"
- source_sentence: 'INC0044420 Dear Team


    Error occurred while processing the EDI transaction


    Interface       micron

    Subsidiary      PIDSMY

    API Name        ext-partners-order-mgmt-papi

    Flow Direction  Inbound

    Source System   SAP

    End System      NA

    File Name       NA

    Storage Path    No Attachment

    Error Source    Mulesoft

    Transaction ID  507b9eb0-b327-11ef-a09a-0a9283444228

    Error Summary   400 BAD_REQUEST

    Error Details   Segment group repeated too many times past end of segment 38 in
    message 1 of interchange 415

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic this is due to user data error more than 10 pair of QTY
    and DTM pairs'
  sentences:
  - "INC0056724 partner portal user cant see their Location (Local Warehouse). see\
    \ attached.\r\naccount name= BRUTAS, CARILO F.\r\nSAP customer code= 5000004741\r\
    \nLocation= CF BRUTAS REF & AIRCON REPAIR SHOP Closed by Caller"
  - "INC0041583 Kindly support create new Partner portal account:\r\nCustomer code:\
    \ 5000028454\r\nEmail:  thanhttbhbr@gmail.com \r\nRefer account:  CS NGUYEN VAN\
    \ THANH\r\nThank you Issue: Create new account\r\nAction Plan taken by AMS: As\
    \ per the request, We have created a partner portal user for this account-Customer\
    \ code: 5000028454\r\nEmail: thanhttbhbr@gmail.com. No actions to be required\
    \ from AMS side. Hence we're good to close the ticket."
  - 'INC0060233 Dear Team


    Error occurred while processing the EDI transaction


    Interface       ETRAVEL Posting

    Subsidiary      PFSAP

    API Name        pana-hriq-mgmt-papi

    Flow Direction  inbound

    Source System   PFSAP SFTP

    End System      SAP-BTP

    File Name       No File Name

    Storage Path    No Attachment

    Error Source    Mulesoft

    Transaction ID  4f8da790-42b6-11f0-a455-ae1abc9b500a

    Error Summary   501 EXPRESSION

    Error Details   Invalid payload

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic user provided invalid payload'
- source_sentence: INC0053873 Pls check PSV-SO-2503005269 & PSV-SO-2503005492 V-invoice
    number has not been responed from SAP to SFDC re-trigger send data from SAP to
    SFDC
  sentences:
  - 'INC0047808 Dear Team


    Error occurred while processing the EDI transaction


    Interface       CONTI AS2 PO

    Subsidiary      PIDSMY

    API Name        ext-partners-order-mgmt-as2-eapi

    Flow Direction  Inbound

    Source System   Continental

    End System      SAP S4Hana

    File Name       Delfor-Mrp

    Storage Path    No Attachment

    Error Source    Mulesoft

    Transaction ID  b19a80d0-d34d-11ef-bd6c-56e8d063249c

    Error Summary   500 TIMEOUT

    Error Details   ***********443/api/v1/PIDSAP/delfor-mrp/continental'' failed:
    Timeout exceeded.

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Upon checking, the forecast report has been updated successfully.
    The error was due to the long processing in the ext-partners-order-mgmt-papi but
    in the background it has completed the processing successfully.'
  - "INC0014533 PACMY-SO-2404003223 missing invoice number Dear user ,\r\n\r\nYour\
    \ incident has now been recorded and worked under problem ticket no. PRB0040032\
    \ for which the issue identified and work around has been pushed over all the\
    \ NSCs.\r\n\r\nKindly refer to the above problem ticket for any update."
  - "INC0042117 PACMY-SO-2410003045 Hi ABU UBAIDAH AMIR AHMAD RAZALI,\r\n\r\nAs per\
    \ your confirmation over teams, we are resolving this ticket."
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
    'INC0053873 Pls check PSV-SO-2503005269 & PSV-SO-2503005492 V-invoice number has not been responed from SAP to SFDC re-trigger send data from SAP to SFDC',
    "INC0047808 Dear Team\n\nError occurred while processing the EDI transaction\n\nInterface       CONTI AS2 PO\nSubsidiary      PIDSMY\nAPI Name        ext-partners-order-mgmt-as2-eapi\nFlow Direction  Inbound\nSource System   Continental\nEnd System      SAP S4Hana\nFile Name       Delfor-Mrp\nStorage Path    No Attachment\nError Source    Mulesoft\nTransaction ID  b19a80d0-d34d-11ef-bd6c-56e8d063249c\nError Summary   500 TIMEOUT\nError Details   ***********443/api/v1/PIDSAP/delfor-mrp/continental' failed: Timeout exceeded.\nComments\n\nNote: This is an automated mail, please do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic Upon checking, the forecast report has been updated successfully. The error was due to the long processing in the ext-partners-order-mgmt-papi but in the background it has completed the processing successfully.",
    'INC0042117 PACMY-SO-2410003045 Hi ABU UBAIDAH AMIR AHMAD RAZALI,\r\n\r\nAs per your confirmation over teams, we are resolving this ticket.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.8125, 0.7084],
#         [0.8125, 1.0000, 0.7093],
#         [0.7084, 0.7093, 1.0000]])
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

* Size: 10,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                           | sentence_1                                                                           | label                                                          |
  |:--------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                               | string                                                                               | float                                                          |
  | details | <ul><li>min: 14 tokens</li><li>mean: 136.39 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 16 tokens</li><li>mean: 140.12 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.51</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>INC0038248 All SO in List Excel is appear In SAP System & Partner Portal, But in SFDC is not appear (Cannot found when we searching), For detail see Excel Attachment. Issue: ISSUE - SO Number Cannot Read (Appear) in SALES FORCES  <br>Action plan taken by AMS: I have given the access to all the list of given so's . For the temporary fix we have raised problem ticket-PRB0040110, Team is working on it. No actions to be required from sfdc side. Hence closing the ticket.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | <code>INC0041974 Dear Team<br><br>Error occurred while processing the EDI transaction<br><br>Interface<br>Subsidiary      No Subsidiary<br>API Name        pana-pagitp-mgmt-eapi<br>Flow Direction  Inbound<br>Source System   PAGITP<br>End System      SAP S4Hana<br>File Name       No File Name<br>Storage Path    No Attachment<br>Error Source    Mulesoft<br>Transaction ID  a6f3ff16-e099-47b3-b157-490d311f4a0c<br>Error Summary   500 SOURCE_RESPONSE_SEND<br>Error Details   Client connection was closed<br>Comments<br><br>Note: This is an automated mail, please do not reply.<br><br>Thanks and Regards,<br><br>APAC Support, Panasonic 1. Checked the log in Cloudwatch.  <br>2. Checked the file in common storage.  <br>3. Verify the data if processed in MBP.  <br>Checked the FPL Invoice ECS4K1881006 is processed succesfully in MBP</code> | <code>1.0</code> |
  | <code>INC0047634 Customer AVNET has highlighted three issues to be resolved. The third one has to be resolved by 16th and all impacted EDI data must be retransmitted (impacted doc list as attached).  <br>1.	Address mismatch on PDF vs EDI data, correct address is “#05-03”, not “#06-03”, please update in EDI data.  <br>2.	Standardized invoice reference format, PDF adopted 10 digits vs EDI data is 9 digits (without leading zero). EDI data has to match with PDF data.  <br>3.	For billing with multiple CPO reference, EDI data is mapping the first CPO reference for all line items. PDF data is correct. Eg: invoice # 230017393 Fix is in progress created change request.</code>                                                                                                                                                                                                                                                                                                                                                                                  | <code>INC0066707 This is linked to incident INC0065971, but in a bigger number or even scope of problems. We had a fiannce review meeting in department heads and Managing Director. We have identified a much bigger gap in the WIP variance between SAP and SFDC. The gap is $1.13 million NZD. I have attached a variance report of the two systems. Could you please prioritise this to the top? We need the investigation ASAP. Let's also have a call to clarify the importance.</code>                                                                                                                                                                                                                                                                                                                                                                    | <code>0.0</code> |
  | <code>INC0043319 Dear Team<br><br>Error occurred while processing the EDI transaction<br><br>Interface       internal-stocks<br>Subsidiary      PAU<br>API Name        pana-sdesk-ext-eapi<br>Flow Direction  outbound<br>Source System   PAU<br>End System      ZOHO<br>File Name       No File Name<br>Storage Path    No Attachment<br>Error Source    pana-sdesk-ext-eapi<br>Transaction ID  ae7e1300-aae7-11ef-8b0c-02b4130d4440<br>Error Summary   500 SOURCE_RESPONSE_SEND<br>Error Details   SOURCE_RESPONSE_SEND: '/Zoho/Live/Outbound/status/remote/SPR_Status_20241121090803362047.csv' cannot be renamed because '/Zoho/Live/Outbound/status/done/SPR_Status_20241121090803362047.csv' already exists<br>Comments<br><br>Note: This is an automated mail, please do not reply.<br><br>Thanks and Regards,<br><br>APAC Support, Panasonic The setup for the file poller in outbound servicedesk is it does not do overwrite, thus, the issue encountered. As a workaround, the previous file has been deleted. As for the resolution, overwrite is now enabled.</code> | <code>INC0025889 User: Yeo Soon  <br>Received a J4U order reply change from factory for P/O# 100024597 \| Migrated P/O 719426  <br>iDOC hit error: Instance 100024597 of object type PurchaseOrder co uld not be changed  <br>Could you kindly help to check and advise. Due to PIDSMY changes Destination city is mandatory in PO. Hence system failed to update the PO. issue is resolved hence closing the ticket.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                        | <code>0.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 12
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
- `num_train_epochs`: 12
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

### Training Logs
| Epoch | Step | Training Loss |
|:-----:|:----:|:-------------:|
| 0.8   | 500  | 0.1959        |
| 1.6   | 1000 | 0.1914        |
| 2.4   | 1500 | 0.1872        |
| 3.2   | 2000 | 0.1814        |
| 4.0   | 2500 | 0.1822        |
| 4.8   | 3000 | 0.1793        |
| 5.6   | 3500 | 0.1786        |
| 6.4   | 4000 | 0.1772        |
| 7.2   | 4500 | 0.1752        |
| 8.0   | 5000 | 0.1768        |
| 8.8   | 5500 | 0.1761        |
| 9.6   | 6000 | 0.1728        |
| 10.4  | 6500 | 0.1749        |
| 11.2  | 7000 | 0.1744        |
| 12.0  | 7500 | 0.1742        |


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