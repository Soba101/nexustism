---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:5000
- loss:MultipleNegativesRankingLoss
base_model: Qwen/Qwen3-Embedding-0.6B
widget:
- source_sentence: 'INC0046802 Dear Team


    Error occurred while processing the EDI transaction


    Interface       Future po reprocess

    Subsidiary      PIDSMY

    API Name        ext-partners-order-mgmt-papi

    Flow Direction  Inbound

    Source System   E-Mail

    End System      SAP S4Hana

    File Name       ERROR_FUTUREELEC_PO_20122420000008.xls

    Storage Path    No Attachment

    Error Source    MuleSoft

    Transaction ID  badcb460-cbff-11ef-91e8-62ef9c19a01e

    Error Summary   500 BUSINESS

    Error Details

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic the is due to recent code change in production ,alr fixed
    and notified user'
  sentences:
  - 'INC0043670 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Interface       GITP

    Subsidiary      PIDSMY

    API Name        inds-global-if-mgmt-papi

    Flow Direction  Outbound

    Source System   SAP

    End System      IBMMQ

    File Name       0000000001724852_4401b2b0-ad22-11ef-bd0c-96a5d4410501.xml

    Storage Path    /INDS/prod/outbound/nocompany/purchase-order/0000000001724852_4401b2b0-ad22-11ef-bd0c-96a5d4410501.xml

    Error Source    Mulesoft

    Transaction ID  4401b2b0-ad22-11ef-bd0c-96a5d4410501

    Error Summary   500 CONNECTIVITY

    Error Details   ***********443/api/v1/aleaud'' failed: Remotely closed.

    Comments        Unable to find 0000000001724852_4401b2b0-ad22-11ef-bd0c-96a5d4410501.xml
    from Backup location


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Verified the PO was successfully transmitted  in GITP'
  - 'INC0064196 Dear Team


    Error occurred while processing the EDI transaction


    Interface       gid-emailid

    Subsidiary      PAPAMY

    API Name        pana-global-hriq-sapi

    Flow Direction  inbound

    Source System

    End System

    File Name       NA

    Storage Path    No Attachment

    Error Source    SuccessFactor-OData API

    Transaction ID  05e9f800-5d7d-11f0-98ec-c29b10320a21

    Error Summary   400 BAD_REQUEST

    Error Details   Error response from SuccessFactors API

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic raised to jane hoh // sj'
  - 'INC0065846 Dear Team


    Error occurred while processing the EDI transaction


    Interface

    Subsidiary      No Subsidiary

    API Name        pana-pagitp-mgmt-eapi

    Flow Direction  Inbound

    Source System   PAGITP

    End System      SAP S4Hana

    File Name       No File Name

    Storage Path    No Attachment

    Error Source    Mulesoft

    Transaction ID  ee55f200-af30-4e06-8587-59da8c3279e6

    Error Summary   500 SOURCE_RESPONSE_SEND

    Error Details   Client connection was closed

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic data verified in sap'
- source_sentence: "INC0056319 User: HONG YEE CHAN/ Mafe\r\nIssue: ePO cancelled and\
    \ posting successful but SAP is locked for editing.\r\n\r\nPlease help to release\
    \ PO. As per user request released the PO. User able to proceed. User confirmed\
    \ to close the ticket."
  sentences:
  - "INC0034880 AT WOLi NO: 00000004, PART ALREADY DELIVERED, BUT WE CAN'T CONSUME\
    \ THE PART. APPEAR ERROR:-\r\n\r\n\"THE CONSUMED QTY CAN'T BE HIGHER THAN THE\
    \ LINE QTY AND CAN'T BE HIGHER THAN THE DELIVERED QTY.\"\r\n\r\nPLEASE REFER TO\
    \ ATTACHMENT. THANKS. Issue: PM-WO-2408000024 ~ PART CANT CONSUME.\r\nAction planned\
    \ taken by AMS:  updated the Consumed qty for PM-WO-2408000024. No action to be\
    \ required from AMS  SFDC side.\r\nHence, closing this ticket."
  - "INC0045707 Kindly Please help for WO PGI-WO-2412000574 is Cannot Generated Invoiced,\
    \ when user try to Generated Invoiced is Appear Error : Occur the error when generated\
    \ the invoiced Investigation done : User was unable to generate invoice due to\
    \ some missing steps.\r\n\r\nAction taken : We have generated invoice on behalf\
    \ of the user and asked user to carry out the next steps."
  - "INC0046784 Dear Team\n\nError occurred while processing the EDI transaction\n\
    \nInterface       SO-REPORT\nSubsidiary      PIDSTH\nAPI Name        ext-partners-order-mgmt-papi\n\
    Flow Direction  Inbound\nSource System   SAP\nEnd System      NA\nFile Name  \
    \     NA\nStorage Path    No Attachment\nError Source    Mulesoft\nTransaction\
    \ ID  b4ab71df-257f-4d7d-946c-e86c9ea279b8\nError Summary   501 RETRY_EXHAUSTED\n\
    Error Details   Path '/INDS/inbound/pidsth/cpo/satl/summary/SATL PIDSTH Summary\
    \ Report_14181903.xls' doesn't exist\nComments\n\nNote: This is an automated mail,\
    \ please do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic Checked\
    \ the log and Mule backup, the CPOs from source file were processed in MBP, only\
    \ the summary/SO reports were not generated after deployment of  Starsuper (due\
    \ to a bug in the fix done for the SR)\r\nHave informed the Sales IT to inform\
    \ user to just check the CPOs directly in MBP"
- source_sentence: "INC0056117 Pls cancel EWS PSV-EWS-2504000295 in SFDC and don't\
    \ trigger to SAP Issue: Cancel EWS PSV-EWS-2504000295 in SFDC\r\n\r\nCategory:\
    \ Configuration\r\n\r\nFailure point: Invalid entry\r\n\r\nRoot cause: Request\
    \ to cancel EWS and prevent SAP trigger\r\n\r\nFix: Cancelled EWS in SFDC and\
    \ verified no SAP trigger\r\n\r\nPreventive Action: Manual verification of EWS\
    \ cancellation\r\n\r\nOutage: NA"
  sentences:
  - INC0042707 Pls cancel EWS PSV-EWS-2411000226 and don't trigger to SAP Cancel EWS
    PSV-EWS-2411000226
  - 'INC0063891 Dear Team


    Error occurred while processing the EDI transaction


    Interface

    Subsidiary      No Subsidiary

    API Name        pana-pagitp-mgmt-eapi

    Flow Direction  Inbound

    Source System   PAGITP

    End System      SAP S4Hana

    File Name       No File Name

    Storage Path    No Attachment

    Error Source    Mulesoft

    Transaction ID  327fe9dd-3e7f-44a2-a2a3-672e28ac20d0

    Error Summary   500 SOURCE_RESPONSE_SEND

    Error Details   Client connection was closed

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic YEMM_IPL_INVOICE01 iDoc posted successfully to SAP. No
    action needed.'
  - 'INC0041960 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Interface       hriq-response

    Subsidiary      PIDSAP

    API Name        pana-hriq-mgmt-papi

    Flow Direction  inbound

    Source System   HRIQ

    End System      eWork SNow System

    File Name       No File Name

    Storage Path    /INDS/prod/inbound/pidsap/daily-allowance/m3dvos411utxyvi2luij_788c1911-5c39-4ae6-a04c-8726d4c7b553.dat

    Error Source    SAP S/4Hana

    Transaction ID  788c1911-5c39-4ae6-a04c-8726d4c7b553

    Error Summary   400 BAD_REQUEST

    Error Details   ***********44300/sap/opu/odata/sap/YMFI_SAPEWORK_API_SRV/inECLAIMSet/''
    failed: bad request (400).eWork Reference Number: eTR-T1-24000301

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Daily Allowance has not been posted in HRIQ since the
    company code is not PIDSAP. Error received is expected'
- source_sentence: 'INC0025622 Need to debug the transaction code ZMFI_COGS_FRDINV
    to get the detailed error message. Resolution notes copied from Parent Incident:
    File got uploaded successfully it is data issue.'
  sentences:
  - 'INC0058939 Dear Team


    Error occurred while processing the EDI transaction


    Interface

    Subsidiary      No Subsidiary

    API Name        pana-pagitp-mgmt-eapi

    Flow Direction  Inbound

    Source System   PAGITP

    End System      SAP S4Hana

    File Name       No File Name

    Storage Path    No Attachment

    Error Source    Mulesoft

    Transaction ID  b5ef4460-38b8-11f0-b4f2-26f4faf8b99a

    Error Summary   500 CONNECTIVITY

    Error Details   Could not establish SFTP connection with host: ''10.86.48.62''
    at port: ''22'' - Session.connect: java.net.SocketTimeoutException: Read timed
    out

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Scheduler runs every 2 mins. No action needed.'
  - 'INC0057481 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Interface       Biometric - Time Clock Data

    Subsidiary      PAPAMY

    API Name        pana-hriq-mgmt-papi

    Flow Direction  inbound

    Source System   SAP ECC FI - FTP

    End System      pana-global-hriq-sapi

    File Name       /HR/BioTime/PRD/ERROR/ERROR_Panasonic 202505070746_3da8c770-2ad4-11f0-9c71-2ef832138abc_1.txt

    Storage Path    /PAPAMY/biometric/time-clock-data/Panasonic 202505070746_3da8c770-2ad4-11f0-9c71-2ef832138abc.txt

    Error Source    Mulesoft

    Transaction ID  3da8c770-2ad4-11f0-9c71-2ef832138abc

    Error Summary   500 RUNTIME_ERROR

    Error Details   ***********443/api/v1/PAPAMY/biometric/time-clock-data'' failed:
    Timeout exceeded.

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Time clock data manually uploaded to SF.'
  - 'INC0054108 Dear Team


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

    Transaction ID  0197ec10-0843-11f0-a511-6a58f71b67be

    Error Summary   500 COMPOSITE_ROUTING

    Error Details   COMPOSITE_ROUTING: Exception(s) were found for route(s): Route
    0: org.mule.runtime.core.api.retry.policy.RetryPolicyExhaustedException: ''until-successful''
    retries exhausted

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic duplicate issue'
- source_sentence: 'INC0044742 Dear Team


    Error occurred while processing the EDI transaction


    Interface       GITP

    Subsidiary      NA

    API Name        inds-global-if-mgmt-papi

    Flow Direction  Inbound

    Source System   IBMMQ

    End System      SAP

    File Name       PIDSAP60670149500000000001_invalidInterface_01_001_101220241423

    Storage Path    No Attachment

    Error Source    Mulesoft

    Transaction ID  ID:363036373031343935303030303030303030303120202020

    Error Summary   400 BAD_REQUEST

    Error Details   Invalid Interface - Unexpected combination of ApplicationIdCode,
    DataTypeCode and EdiFormatType in GITP header info

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic this is data issue from source system, we can safely omit
    this issue'
  sentences:
  - 'INC0045839 Dear Team


    Error occurred while processing the EDI transaction


    Interface       SO-REPORT

    Subsidiary      PIDSMY

    API Name        ext-partners-order-mgmt-papi

    Flow Direction  Inbound

    Source System   SAP

    End System      NA

    File Name       NA

    Storage Path    No Attachment

    Error Source    Mulesoft

    Transaction ID  a3347b2e-ee45-4507-8921-0dbaf9380920

    Error Summary   500 UNKNOWN

    Error Details   Exception was found writing to file ''/INDS/inbound/pidsmy/cpo/sumitronics/so/SUMITRONICS_SO
    Report_08591283_.xls''

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic PIDSMY_SUMITRONICS_SAPINDS_CPO REPORT has already sent
    to user, this is duplicate call from SAP'
  - "INC0053385 Dear Team\n\nError occurred while processing the EDI transaction\n\
    \nInterface       GSSI\nSubsidiary      No Subsidiary\nAPI Name        inds-pagitp-sapi\n\
    Flow Direction  Outbound\nSource System   SAP\nEnd System      PA-GITP\nFile Name\
    \       FID20701_SalesResult_GSSI_00029914_2025031601241533000\nStorage Path \
    \   No Attachment\nError Source    SEGLink-SFTP\nTransaction ID  07997ab2-812f-4396-ba2a-58d46f8f8261\n\
    Error Summary\nError Details   Could not establish SFTP connection with host:\
    \ '10.86.48.62' at port: '22' - timeout: socket is not established\nComments\n\
    \nNote: This is an automated mail, please do not reply.\n\nThanks and Regards,\n\
    \nAPAC Support, Panasonic Scheduler failed due to Firewall Technical Refresh Activity\r\
    \nNew files were already transmitted from Monday morning so have confirmed with\
    \ SEGLINK that Sunday files don't need to resend"
  - 'INC0042445 Dear Team


    Error occurred while processing the EDI transaction


    Interface       PANA.ACCTEVENTS-PRD.GENESIS.Q

    Subsidiary      PSV

    API Name        pana-sf-mc-sapi

    Flow Direction  inbound

    Source System   PAPI

    End System      genesis

    File Name       NA

    Storage Path    No Attachment

    Error Source    Mulesoft

    Transaction ID  922355a0-a338-11ef-b5d2-4ae2ffd96225

    Error Summary   400 RETRY_EXHAUSTED

    Error Details   RETRY_EXHAUSTED: Invalid status code: 400, response body:

    Bad Message 400


    reason: Bad Request


    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic this case is due to Salesforce service down on 15th Nov,
    the service has resumed normally'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on Qwen/Qwen3-Embedding-0.6B

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B). It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) <!-- at revision c54f2e6e80b2d7b7de06f51cec4959f6b3e03418 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 1024 dimensions
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
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': True, 'include_prompt': True})
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
queries = [
    "INC0044742 Dear Team\n\nError occurred while processing the EDI transaction\n\nInterface       GITP\nSubsidiary      NA\nAPI Name        inds-global-if-mgmt-papi\nFlow Direction  Inbound\nSource System   IBMMQ\nEnd System      SAP\nFile Name       PIDSAP60670149500000000001_invalidInterface_01_001_101220241423\nStorage Path    No Attachment\nError Source    Mulesoft\nTransaction ID  ID:363036373031343935303030303030303030303120202020\nError Summary   400 BAD_REQUEST\nError Details   Invalid Interface - Unexpected combination of ApplicationIdCode, DataTypeCode and EdiFormatType in GITP header info\nComments\n\nNote: This is an automated mail, please do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic this is data issue from source system, we can safely omit this issue",
]
documents = [
    'INC0042445 Dear Team\n\nError occurred while processing the EDI transaction\n\nInterface       PANA.ACCTEVENTS-PRD.GENESIS.Q\nSubsidiary      PSV\nAPI Name        pana-sf-mc-sapi\nFlow Direction  inbound\nSource System   PAPI\nEnd System      genesis\nFile Name       NA\nStorage Path    No Attachment\nError Source    Mulesoft\nTransaction ID  922355a0-a338-11ef-b5d2-4ae2ffd96225\nError Summary   400 RETRY_EXHAUSTED\nError Details   RETRY_EXHAUSTED: Invalid status code: 400, response body:\nBad Message 400\n\nreason: Bad Request\n\nComments\n\nNote: This is an automated mail, please do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic this case is due to Salesforce service down on 15th Nov, the service has resumed normally',
    "INC0053385 Dear Team\n\nError occurred while processing the EDI transaction\n\nInterface       GSSI\nSubsidiary      No Subsidiary\nAPI Name        inds-pagitp-sapi\nFlow Direction  Outbound\nSource System   SAP\nEnd System      PA-GITP\nFile Name       FID20701_SalesResult_GSSI_00029914_2025031601241533000\nStorage Path    No Attachment\nError Source    SEGLink-SFTP\nTransaction ID  07997ab2-812f-4396-ba2a-58d46f8f8261\nError Summary\nError Details   Could not establish SFTP connection with host: '10.86.48.62' at port: '22' - timeout: socket is not established\nComments\n\nNote: This is an automated mail, please do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic Scheduler failed due to Firewall Technical Refresh Activity\r\nNew files were already transmitted from Monday morning so have confirmed with SEGLINK that Sunday files don't need to resend",
    "INC0045839 Dear Team\n\nError occurred while processing the EDI transaction\n\nInterface       SO-REPORT\nSubsidiary      PIDSMY\nAPI Name        ext-partners-order-mgmt-papi\nFlow Direction  Inbound\nSource System   SAP\nEnd System      NA\nFile Name       NA\nStorage Path    No Attachment\nError Source    Mulesoft\nTransaction ID  a3347b2e-ee45-4507-8921-0dbaf9380920\nError Summary   500 UNKNOWN\nError Details   Exception was found writing to file '/INDS/inbound/pidsmy/cpo/sumitronics/so/SUMITRONICS_SO Report_08591283_.xls'\nComments\n\nNote: This is an automated mail, please do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic PIDSMY_SUMITRONICS_SAPINDS_CPO REPORT has already sent to user, this is duplicate call from SAP",
]
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# [1, 1024] [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[0.9923, 0.9922, 0.9918]])
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
  | details | <ul><li>min: 14 tokens</li><li>mean: 147.66 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 16 tokens</li><li>mean: 150.13 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.48</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | label            |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>INC0038234 Issues found on eTravel with no. eTR-M1-24000167:   <br>  <br>1) Print preview issue for travel detail-the date and month for TRIP-01 and TRIP-02 is different.  <br>2) Payment Summary- Net payable to staff/ Company and Pay through Finance (yellow highlighted, the amount should take pre amount but have capture post amount)  <br>3) The exchange rate for pre-travel and post travel. Exchange rate on (Pre) should be the e-travel document creation date and exchange rate on (Post) should display the departure. Pls advise.   <br>Refer to the attachment. New configuration was deployed to handle this issue</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | <code>INC0032817 pls investigate why certain Exchange rates are not transmitted from SAP to ework. Refer to parent ticket INC0030025   <br>  <br>Requester : Mr.Lee Hansoo(lee.hansoo@kr.panasonic.com)  <br>  <br>We are currently entering a total of 12 types of exchange rate information into SAP. However, the exchange rates for CNY-KRW, VND-KRW, and USD-100JPY are not being entered into eWORK.  <br>We would appreciate your prompt resolution of this issue and request that you also input the previously omitted exchange rates in bulk.  <br>Thank you for your attention to this matter. Hello @Lee Hansoo (???)San,  <br>  <br>We have checked in SAP -VND and CNY currency is not config to send to Ework side.  <br>So please raise SR ticket then will do the maintain for required currency.  <br>So I’m proceeding for the closure of this incident create new Sr and share with us.</code>                                                                                                                                                                                                                          | <code>1.0</code> |
  | <code>INC0052496 Dear Team<br><br>Error occurred while processing the EDI transaction. Please find the details below and attached is the file associated to the transaction.<br><br>Flow Direction  outbound<br>Subsidiary      PAPVN-TL2<br>Source System   SAP<br>End System      einvoice<br>File Name       0000000000049977<br>Storage Path    /inbound/SAP/eInvoice/0000000000049977_To_einvoice.json<br>API Name        sgst-fi-invoice-papi<br>Error Source    INVOICE<br>Transaction ID  52e41a20-fa6b-11ef-ad6e-4efcac8a9e54<br>Error Summary   500 HTTP POST on resource 'https://sgst-fi-invoice-einvoice-sapi-qoq0kf.internal-hnygb7.sgp-s1.cloudhub.io:443/api/v1/create-invoice' failed: internal server error (500).<br>Error Details   HTTP POST on resource 'https://sgst-fi-invoice-einvoice-sapi-qoq0kf.internal-hnygb7.sgp-s1.cloudhub.io:443/api/v1/create-invoice' failed: internal server error (500).<br>Comments        Unable to retrive file from Backup location<br><br>Note: This is an automated mail, please do not reply.<br><br>Thanks and Regards,<br><br>APAC Support, Pa...</code> | <code>INC0054696 We would like to report an issue regarding t-code MB90. When executing the t-code, the message prompt says "no message for initial/repeat/error processing exists" for below Material Documents under SLOC M274.  <br>  <br>Output Type: ZGI1  <br>Material Documents:  <br>4900324150  <br>4900324128  <br>4900324129  <br>  <br>Note: We usually encounter this kind of error for newly maintained Storage Location in PPH. Kindly check for any configuration issues. Also, using Material Documents from other/old Storage Location are working fine. Issue:  <br>Output from Goods Movement (MB90) Issues  <br>  <br>Category:  <br>SAP  <br>  <br>Failure point:  <br>No message for initial/repeat/error processing  <br>  <br>Root cause:  <br>Condition record not maintained for new storage location  <br>  <br>Fix:  <br>Maintained ZGI1 Condition Record on Plant 7964 and Storage Location M274 via t-code MN21  <br>  <br>Preventive Action:  <br>Used material documents from other/old storage locations  <br>  <br>Outage:  <br>NA  <br>  <br>If you need any further assistance or modifications, feel free to ask!</code> | <code>0.0</code> |
  | <code>INC0017130 The Email you have just triggered is having invalid or wrong subject.  <br>Please Use valid subject and Resend  <br>  <br>Case:   <br>  <br>User using a different mail client to send email  <br>User is also using an invalid subject,  <br>However, the error email is not sent back to the user to notify the cause  <br>  <br>Cause:   <br> User using a different mail client to send email  <br>the format of the email attribute changes ,  <br>  <br>Outlook email client  <br>fromAddresses=[Prudhvi Raj <prudhvi.raj@sg.panasonic.com>]  <br>  <br>User email client  <br>fromAddresses=[prudhvi.raj@sg.panasonic.com] code updated and deployed, will monitor</code>                                                                                                                                                                                                                                                                                                                                                                                                                                        | <code>INC0050578 ePO001452 - it was RFIed. Afterwhich , the routing section has disappeared. Hence I can't set any routing. Please help urgently Dear @Siew Shien Fu san,   <br>Good evening!!!  <br>  <br>ePO routing fixes during revised stage has been moved into PA production environment.  <br>Hence confirmed with your end, we are proceeding to close this ticket.  <br>  <br>Regards,  <br>Manoj.BK  <br>IT Consultant  <br>Panasonic Information</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | <code>0.0</code> |
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
- `num_train_epochs`: 4
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
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.4984 | 156  | -             |
| 0.9968 | 312  | -             |
| 1.0    | 313  | -             |
| 1.4952 | 468  | -             |
| 1.5974 | 500  | 2.7343        |
| 1.9936 | 624  | -             |
| 2.0    | 626  | -             |
| 2.4920 | 780  | -             |
| 2.9904 | 936  | -             |
| 3.0    | 939  | -             |
| 3.1949 | 1000 | 2.5792        |
| 3.4888 | 1092 | -             |
| 3.9872 | 1248 | -             |
| 4.0    | 1252 | -             |
| 0.4984 | 156  | -             |
| 0.9968 | 312  | -             |
| 1.0    | 313  | -             |
| 1.4952 | 468  | -             |
| 1.5974 | 500  | 2.7951        |
| 1.9936 | 624  | -             |
| 2.0    | 626  | -             |
| 2.4920 | 780  | -             |
| 2.9904 | 936  | -             |
| 3.0    | 939  | -             |
| 3.1949 | 1000 | 2.7611        |
| 3.4888 | 1092 | -             |
| 3.9872 | 1248 | -             |
| 4.0    | 1252 | -             |
| 0.4984 | 156  | -             |
| 0.9968 | 312  | -             |


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