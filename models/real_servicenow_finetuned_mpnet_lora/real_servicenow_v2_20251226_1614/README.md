---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:5000
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: "INC0013984 Below 2 vendors purchasing group data missing in attach\
    \ I/F file but user could check the related information in MM03&ME13, please help\
    \ to check reason asap as related PO couldn’t be issued now.\r\n\r\nVendor 1:\
    \ 21024568CC\r\nVendor 2: 21024568CD\r\n\r\nFile path?/SAP-IF/RECEIVE/FI/Other_Vendor_*.txt\
    \ Explained to user logic of Purchasing Group"
  sentences:
  - 'INC0060777 Dear Team


    Error occurred while processing the EDI transaction


    Interface

    Subsidiary      PIDSTH

    API Name        ext-partners-order-mgmt-papi

    Flow Direction  Inbound

    Source System   SAP

    End System      NA

    File Name       NA

    Storage Path    No Attachment

    Error Source    Mulesoft

    Transaction ID  4b00e1a2-ec16-4018-accd-53017e7339a7

    Error Summary   500 ILLEGAL_PATH

    Error Details   Path ''/INDS/inbound/pidsth/cpo/meta/so/META_SO Report_14094765_.xls''
    doesn''t exist

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Have checked that the batch was successfully processed
    in MBP and the email report was sent to the users'
  - "INC0036864 User unable to create Non Trade PO, prompting error: 'Supplier NM003800\
    \ has not been created for purchasing organization 7230.'\r\nAs per checking,\
    \ NM003800 is extended 7230 but not sure why it displays this error upon creation.\
    \ \r\nPls advise. Rajeswary Ramadas\r\n confirmed to close no action pending from\
    \ our side"
  - "INC0059517 DO Posting failed \r\n\r\nKindey help to check eDO posting failed\
    \ to SAP\r\nD/O 455014577 // eDO-T1-25003669\r\nD/O 455014874 // eDO-T1-25003671\
    \ issue:\r\nFalled to post eDO from eWork\r\n\r\nCategory:\r\nSAP MM\r\n\r\nFailure\
    \ point:\r\nDeficit sales order stock in storage location Z601\r\n\r\nRoot cause:\r\
    \nUser transferred the stock from storage location Z601 to D110 . Hence there\
    \ is no stock in Z601\r\nFix:\r\nNA\r\nPreventive Action:\r\nCheck the stock in\
    \ storage location before posting ."
- source_sentence: 'INC0057519 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Interface       MELCO PO

    Subsidiary      PIDSTH

    API Name        ext-partners-order-mgmt-papi

    Flow Direction  Inbound

    Source System   E-Mail

    End System      SAP S4Hana

    File Name       Copy of MEI_for india_MTT25630068_2_DATA.xls

    Storage Path    /INDS/inbound/pidsth/cpo/melco/backup/Copy of MEI_for india_MTT25630068_2_DATA_a90072c0-2aed-11f0-8c4f-76fd2b8697ae.xls

    Error Source    Mulesoft

    Transaction ID  a90072c0-2aed-11f0-8c4f-76fd2b8697ae

    Error Summary   501 EXPRESSION

    Error Details   Expression Error Occured

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic file sent by user has a data issue'
  sentences:
  - 'INC0042160 Dear Team


    Error occurred while processing the EDI transaction


    Interface       IPS43

    Subsidiary      GITP

    API Name        inds-global-if-mgmt-papi

    Flow Direction  outbound

    Source System   GITP

    End System      SAP

    File Name       No File Name

    Storage Path    No Attachment

    Error Source    SAP

    Transaction ID  3ec60011-a194-11ef-82bf-7e45c3d0ddf1

    Error Summary   500 INTERNAL_SERVER

    Error Details   UNKNOWN: Cannot process event as "inds-global-if-mgmt-papi-scheduler-impl-ips43Flow"
    is stopped

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Upon checking the flow is now set to started since the
    init-state.scheduler is set to started in runtime properties and the flow is enabled
    in  the schedulers. Unable to find in audit logs when the runtime properties or
    schedulers were updated. Issue is no longer happening'
  - 'INC0068949 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Interface       confirmed-bill-of-lading

    Subsidiary      PIDSKR

    API Name        ext-partners-3pl-mgmt-papi

    Flow Direction  inbound

    Source System   3PL

    End System      SAP S4Hana

    File Name       KMTC5SRX20250901114040

    Storage Path    /INDS/inbound/pidskr/stock-reconciliation/KMTC5SRX20250901114040_666397b0-86de-11f0-a1c0-424508f363ff.txt

    Error Source    Mulesoft

    Transaction ID  666397b0-86de-11f0-a1c0-424508f363ff

    Error Summary   501 RETRY_EXHAUSTED

    Error Details   Path ''/pidskr/kr_kmtc/ctl_l/KMTC5SRX20250901114040.RC'' doesn''t
    exist

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic no control file uploaded before uploading the data file,
    reprocessed'
  - 'INC0060872 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Flow Direction

    Subsidiary

    Source System   SAP

    End System      EPRO

    File Name       Y0GMM_ZAO0110_R41_ID20_20250613090011

    Storage Path

    API Name        sgst-audit-papi

    Error Source    PCS

    Transaction ID  8d1ef207-12fc-454c-b4b0-7955aefefb42

    Error Summary   500 Exception was found writing to file ''/outbound/pmi/epro/purchase-orders/Y0GMM_ZAO0110_R41_ID20_20250613090011_8d1ef207-12fc-454c-b4b0-7955aefefb42.txt''

    Error Details   Exception was found writing to file ''/outbound/pmi/epro/purchase-orders/Y0GMM_ZAO0110_R41_ID20_20250613090011_8d1ef207-12fc-454c-b4b0-7955aefefb42.txt''

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Error writing to PCS backup server. No action needed since
    file was posted to GITP'
- source_sentence: "INC0029640 Mr. Sasitar will help to look into the issue after\
    \ discussion today. Hi Pang,\r\n\r\nBased on your confirmation in Teams, we are\
    \ closing this ticket."
  sentences:
  - 'INC0061395 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Flow Direction  inbound

    Subsidiary

    Source System   Service Now

    End System      BASIS

    File Name       NA

    Storage Path    NA

    API Name        ework-snow-mgmt-eapi

    Error Source    SAP

    Transaction ID  b890f99e-4545-4b99-9d13-75895d7d83ca

    Error Summary   400 com.sap.conn.jco.JCoException: (104) JCO_ERROR_SYSTEM_FAILURE:
    SQL error SQL code: 30036 occurred while accessing table USR02. (Remote shortdump:
    DBSQL_SQL_ERROR in system [PNQ|paipnq.asia.gds.panasonic.com|26])

    Error Details   com.sap.conn.jco.JCoException: (104) JCO_ERROR_SYSTEM_FAILURE:
    SQL error SQL code: 30036 occurred while accessing table USR02. (Remote shortdump:
    DBSQL_SQL_ERROR in system [PNQ|paipnq.asia.gds.panasonic.com|26])

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Network issue. Subsequent password requests were successful'
  - "INC0054986 Dear Team\n\nError occurred while processing the EDI transaction\n\
    \nInterface       GITP\nSubsidiary      PIDSMY\nAPI Name        inds-global-if-mgmt-papi\n\
    Flow Direction  Inbound\nSource System   IBMMQ\nEnd System      SAP\nFile Name\
    \       NA\nStorage Path    No Attachment\nError Source    Mulesoft\nTransaction\
    \ ID  ID:363039393836373230303030303030303030303320202020\nError Summary   500\
    \ CONNECTIVITY\nError Details   ***********443/api/v1/sender-receiver-code-reference-data'\
    \ failed: Remotely closed.\nComments\n\nNote: This is an automated mail, please\
    \ do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic 1. Checked the\
    \ log in Cloudwatch.\r\n2. Checked the file in common storage.\r\n3. Have verified\
    \ that the data was processed in MBP."
  - 'INC0053997 Dear Team


    Error occurred while processing the EDI transaction


    Interface       PANA.CART-PRD.MARKETING.Q

    Subsidiary      PSV

    API Name        pana-sf-mc-sapi

    Flow Direction  inbound

    Source System   PAPI

    End System      Salesforce Marketing Cloud

    File Name       NA

    Storage Path    No Attachment

    Error Source    Mulesoft

    Transaction ID  e8fc26a0-06e2-11f0-a511-6a58f71b67be

    Error Summary   500 COMPOSITE_ROUTING

    Error Details   COMPOSITE_ROUTING: Exception(s) were found for route(s): Route
    0: org.mule.runtime.core.api.retry.policy.RetryPolicyExhaustedException: ''until-successful''
    retries exhausted

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic duplicate issue'
- source_sentence: "INC0060813 The order PSV-SO-2505003416 has been invoiced and updated\
    \ in SAP but there is no information in the online part sale invoice. Please check\
    \ and fix it. Thank you Issue:\r\nMissing data in online part sale invoice for\
    \ PSV-SO-2505003416\r\n\r\nCategory:\r\nUser Error\r\n\r\nFailure Point:\r\nOnline\
    \ invoice not updated post SAP billing\r\n\r\nRoot Cause:\r\nSales Order had an\
    \ incompletion log due to unregistered or incorrect model in SAP\r\n\r\nFix:\r\
    \nPSV team registered the correct model or updated the default model in the Sales\
    \ Order to resolve the incompletion log\r\n\r\nPreventive Action:\r\nFollow-up\
    \ with PSV local IT and SAP teams; interim communication and guidance shared with\
    \ stakeholders\r\n\r\nOutage:\r\nNA"
  sentences:
  - "INC0052986 Dear Team\n\nError occurred while processing the EDI transaction.\
    \ Please find the details below and attached is the file associated to the transaction.\n\
    \nInterface       GITP\nSubsidiary      PIDSAP\nAPI Name        inds-global-if-mgmt-papi\n\
    Flow Direction  Outbound\nSource System   SAP\nEnd System      IBMMQ\nFile Name\
    \       0000000002422127_b9024fe0-ff14-11ef-a2e0-daed00c1551a.xml\nStorage Path\
    \    /INDS/prod/outbound/nocompany/ordrsp/0000000002422127_b9024fe0-ff14-11ef-a2e0-daed00c1551a.xml\n\
    Error Source    Mulesoft\nTransaction ID  b9024fe0-ff14-11ef-a2e0-daed00c1551a\n\
    Error Summary   500 CONNECTIVITY\nError Details   ***********443/api/v1/aleaud'\
    \ failed: Remotely closed.\nComments        Unable to find 0000000002422127_b9024fe0-ff14-11ef-a2e0-daed00c1551a.xml\
    \ from Backup location\n\nNote: This is an automated mail, please do not reply.\n\
    \nThanks and Regards,\n\nAPAC Support, Panasonic 1. Check the IDOC/GIS file from\
    \ backup folder in common storage\r\n2. Cross-check the count of YBE1 output from\
    \ MBP\r\n3. Cross-check the data in GITP using the BELNR (PO#) from IDOC file\r\
    \n4. Data is in GITP, no action needed"
  - 'INC0060591 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Flow Direction

    Subsidiary

    Source System   SAP

    End System      SALES_EDI

    File Name       PLAP_INVOICE_ID20_20250611120110

    Storage Path

    API Name        sgst-audit-papi

    Error Source    PCS

    Transaction ID  cc2d2f98-ad9d-4b05-981c-74bbba5cd6fd

    Error Summary   500 Exception was found writing to file ''/outbound/pmi/sales_edi/fpl/PLAP_INVOICE_ID20_20250611120110_cc2d2f98-ad9d-4b05-981c-74bbba5cd6fd.txt''

    Error Details   Exception was found writing to file ''/outbound/pmi/sales_edi/fpl/PLAP_INVOICE_ID20_20250611120110_cc2d2f98-ad9d-4b05-981c-74bbba5cd6fd.txt''

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Error writing to PCS backup server. No action needed since
    file was posted to GITP'
  - 'INC0061236 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Flow Direction

    Subsidiary

    Source System   SAP

    End System      EPRO

    File Name       Y0GMM_ZAO0110_R21_ID20_20250617140015

    Storage Path

    API Name        sgst-audit-papi

    Error Source    PCS

    Transaction ID  48012afe-7d44-4165-a487-67deae4590ef

    Error Summary   500 Exception was found writing to file ''/outbound/pmi/epro/purchase-orders/Y0GMM_ZAO0110_R21_ID20_20250617140015_48012afe-7d44-4165-a487-67deae4590ef.txt''

    Error Details   Exception was found writing to file ''/outbound/pmi/epro/purchase-orders/Y0GMM_ZAO0110_R21_ID20_20250617140015_48012afe-7d44-4165-a487-67deae4590ef.txt''

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Error writing to PCS backup server. No action needed since
    file was posted to GITP'
- source_sentence: "INC0053184 Dear Team\n\nError occurred while processing the EDI\
    \ transaction\n\nInterface       avnet\nSubsidiary      PIDSMY\nAPI Name     \
    \   ext-partners-order-mgmt-papi\nFlow Direction  Inbound\nSource System   SAP\n\
    End System      NA\nFile Name       NA\nStorage Path    No Attachment\nError Source\
    \    Mulesoft\nTransaction ID  c5ebd980-002a-11f0-b207-6eba885a7792\nError Summary\
    \   500 CONNECTIVITY\nError Details   ***********443/api/v1/stock-code/YEL' failed:\
    \ Remotely closed.\nComments\n\nNote: This is an automated mail, please do not\
    \ reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic Root Cause: Timeout\
    \ exceeded and caused by 100% CPU utilization.\r\nChecked in Mule backup and cross-check\
    \ the failed PO\r\nVerify if the failed PO is included in the processed POs\r\n\
    Checked the failed PO is not in the summary report/AS2 flat file, reprocessed\
    \ the failed PO via postman"
  sentences:
  - 'INC0049616 Dear Team


    Error occurred while processing the EDI transaction


    Interface       GITP

    Subsidiary      NA

    API Name        inds-global-if-mgmt-papi

    Flow Direction  Inbound

    Source System   IBMMQ

    End System      SAP

    File Name       PIDSAP60822026700000000001_invalidInterface_01_001_050220250213

    Storage Path    No Attachment

    Error Source    Mulesoft

    Transaction ID  ID:363038323230323637303030303030303030303120202020

    Error Summary   400 BAD_REQUEST

    Error Details   Invalid Interface - Unexpected combination of ApplicationIdCode,
    DataTypeCode and EdiFormatType in GITP header info

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Data issues are PO from PIDSKR to PIDSAP - to ignore'
  - "INC0055286 For customer TTI (sold to: 80963700 & 80947847 & 80931356), for erroneous\
    \ CPO, user manually creates SO in SAP. Can the erroneous CPO reprocessed via\
    \ ZMSD_II0024 or can MuleSoft generate an error email for user to resend to MuleSoft?\
    \ Please help to check From: Tanisha Rajah <tanisha.rajah@my.panasonic.com> \r\
    \nSent: Monday, June 23, 2025 7:32 AM\r\nTo: Salvacion Costales <salvaciong.costales@sg.panasonic.com>;\
    \ Venkatesh Kuna <venkatesh.kuna@sg.panasonic.com>\r\nCc: Balakumar Ganesan <balakumar.ganesan@sg.panasonic.com>;\
    \ PISCAP_SAP_SD_BRS <sap-sd-brs-ams@sg.panasonic.com>; PIDSMY Business Process\
    \ <business.process@my.panasonic.com>; Ramanjaneyulu Reddy Bandi <ramanjaneyulu.bandi@sg.panasonic.com>\r\
    \nSubject: RE: Incident -INC0055286\r\n\r\nDear Sally,\r\n\r\nNoted with thanks.\
    \ We will proceed to use. You may close the incident ticket\r\n\r\nBest Regards,\r\
    \nTanisha Thasaratharajah\r\n \r\n********************************************************\r\
    \nInformation System Group\r\nPanasonic Industrial Devices Sales (M) Sdn. Bhd.\
    \ (PIDSMY)\r\nMobile Num: 0122489846\r\nEmail: tanisha.rajah@my.panasonic.com\r\
    \n********************************************************"
  - "INC0029177 User : Khine \r\n\r\nPO : 100043847 \r\n\r\nEDI  Order reply J4U \
    \ idoc received on 24 Jul didn't get process . \r\nPls check and advise . For\
    \ the PO 100043847 another idoc failed with  J3U with over quantity. Hence system\
    \ not allowed to process J4U idoc. Informed to the user."
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
    "INC0053184 Dear Team\n\nError occurred while processing the EDI transaction\n\nInterface       avnet\nSubsidiary      PIDSMY\nAPI Name        ext-partners-order-mgmt-papi\nFlow Direction  Inbound\nSource System   SAP\nEnd System      NA\nFile Name       NA\nStorage Path    No Attachment\nError Source    Mulesoft\nTransaction ID  c5ebd980-002a-11f0-b207-6eba885a7792\nError Summary   500 CONNECTIVITY\nError Details   ***********443/api/v1/stock-code/YEL' failed: Remotely closed.\nComments\n\nNote: This is an automated mail, please do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic Root Cause: Timeout exceeded and caused by 100% CPU utilization.\r\nChecked in Mule backup and cross-check the failed PO\r\nVerify if the failed PO is included in the processed POs\r\nChecked the failed PO is not in the summary report/AS2 flat file, reprocessed the failed PO via postman",
    "INC0029177 User : Khine \r\n\r\nPO : 100043847 \r\n\r\nEDI  Order reply J4U  idoc received on 24 Jul didn't get process . \r\nPls check and advise . For the PO 100043847 another idoc failed with  J3U with over quantity. Hence system not allowed to process J4U idoc. Informed to the user.",
    'INC0055286 For customer TTI (sold to: 80963700 & 80947847 & 80931356), for erroneous CPO, user manually creates SO in SAP. Can the erroneous CPO reprocessed via ZMSD_II0024 or can MuleSoft generate an error email for user to resend to MuleSoft? Please help to check From: Tanisha Rajah <tanisha.rajah@my.panasonic.com> \r\nSent: Monday, June 23, 2025 7:32 AM\r\nTo: Salvacion Costales <salvaciong.costales@sg.panasonic.com>; Venkatesh Kuna <venkatesh.kuna@sg.panasonic.com>\r\nCc: Balakumar Ganesan <balakumar.ganesan@sg.panasonic.com>; PISCAP_SAP_SD_BRS <sap-sd-brs-ams@sg.panasonic.com>; PIDSMY Business Process <business.process@my.panasonic.com>; Ramanjaneyulu Reddy Bandi <ramanjaneyulu.bandi@sg.panasonic.com>\r\nSubject: RE: Incident -INC0055286\r\n\r\nDear Sally,\r\n\r\nNoted with thanks. We will proceed to use. You may close the incident ticket\r\n\r\nBest Regards,\r\nTanisha Thasaratharajah\r\n \r\n********************************************************\r\nInformation System Group\r\nPanasonic Industrial Devices Sales (M) Sdn. Bhd. (PIDSMY)\r\nMobile Num: 0122489846\r\nEmail: tanisha.rajah@my.panasonic.com\r\n********************************************************',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.5984, 0.6287],
#         [0.5984, 1.0000, 0.5124],
#         [0.6287, 0.5124, 1.0000]])
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
  | details | <ul><li>min: 17 tokens</li><li>mean: 134.23 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 16 tokens</li><li>mean: 137.31 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.48</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                   | label            |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>INC0047164 Pls check we cannot generate invoice in SFDC update holiday for PSV</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | <code>INC0011982 Check eWork issue Vendor master does not exist</code>                                                                                                                                                                                                                                                                                                                                                                       | <code>0.0</code> |
  | <code>INC0052612 CBL partially posted (only GR) for billing no: 230025875. Document is locked, please advise how to proceed re-ran / reprocess and resolved it. However there is another ticket raised by business which is being checked technically for any improvement</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | <code>INC0019062 We have been noticed CS Thai Ha 2 account several times displayed as ""CS THAI HA 2 m?i Thái Hà"" which is in completely wrong form.  <br>Please support us to correct it to fixed name ""CS THAI HA 2"" We have updated the ASC name from CS THAI HA 2 m?i Thái Hà to CS THAI HA 2. No actions to be required from sf side, hence closing.  <br>Please feel free to raise another request, if you face further issue.</code> | <code>1.0</code> |
  | <code>INC0035106 Our BI analysts rely on connecting to the PISCAP SFTP folder via our server to extract data for BI reports.   <br>This connection is currently unavailable (error msg attached).  <br>  <br>Service 202.68.210.62  <br>Socket Error Code: Timed Out  <br>Msg: A connection attempt failed because the connected party did not properly respond after a period of time, or established connection failed because connected host has failed to respond  <br>  <br>Management has been asking for an update on this issue for days now, and we are struggling to provide a satisfactory explanation for the delay.  <br>  <br>Also, the SFTP folder is not being updated anymore, no new files there since the 4/Sep  <br>Would this be related to the issues in connecting our server to SFTP?  <br>  <br>Thank you This issue was discussed and resolved via Teams' GC/emails  <br>Summary:  <br>There were 2 issues, and both are already resolved:  <br>1. Unable to connect to SFTP from Hussmann Network  <br>     Resolve after IP addresses are whitelisted by Network team  <br>2. Files we...</code> | <code>INC0037508 [OTC] Retrigger SO number 9712389204 to eWork User accidentally deleted in ESO so we help retrigger using Y0NSD_0102</code>                                                                                                                                                                                                                                                                                                 | <code>0.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `num_train_epochs`: 4
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
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
- `num_train_epochs`: 4
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
| Epoch | Step |
|:-----:|:----:|
| 1.0   | 79   |
| 2.0   | 158  |
| 3.0   | 237  |
| 4.0   | 316  |
| 1.0   | 79   |
| 2.0   | 158  |
| 3.0   | 237  |
| 4.0   | 316  |
| 1.0   | 79   |
| 2.0   | 158  |
| 3.0   | 237  |
| 4.0   | 316  |


### Framework Versions
- Python: 3.11.14
- Sentence Transformers: 5.2.0
- Transformers: 4.57.3
- PyTorch: 2.11.0.dev20251223+cu128
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