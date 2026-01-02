---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:2500
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: "INC0035257 User: Heyne\r\nIssue: User wants to create invoice\
    \ and requires change of GL Recon Account Code. Draft saved INV-001765.\r\nGL\
    \ Recon Account Code Default to 23010002 (Sundry Payable - Non Conso) and doesn't\
    \ allow to edit. \r\nPlease find previous eIV-S1-24000824 raised for 23730002\
    \ (Sundry Payable - Conso) Mandatory fields needed to be filled before GL acc\
    \ field could be edited. User advised to do so which resolved the issue"
  sentences:
  - "INC0067846 Jian Yang (back up PIC) said he didn't receive any of eSO. Customer\
    \ had chased already. Dear All,\r\nAs discussed the delegation is working as expected\
    \ and as clarification is provided and upon confirmation we are proceeding to\
    \ close the ticket.\r\nRegards\r\nAseem.S"
  - INC0032219 Kindly Please check All Part In PGI-SO-2405004314 status already Received
    In Partner Portal and Sales Forces (SFDC), but in SAP System status is not Received,
    Please Help to Check And Syncronize this, for detail please see ServiceNow attachment
    Picture Synced the sale order status with SAP
  - "INC0041055 Dear AMS,\r\nPAPVN reverted 2 orders no. 100840779, 100840788, but\
    \ COGI of ARBDCM300171 is not clear.\r\nPlant: 11P1.\r\n\r\nPLease help to check\
    \ it.\r\n\r\nBrgs COGI has been deleted after check all data reversal and make\
    \ sure no hanging transaction for those 2 prod order."
- source_sentence: 'INC0054076 Dear Team


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

    Transaction ID  d9b79aa1-07bc-11f0-a511-6a58f71b67be

    Error Summary   500 COMPOSITE_ROUTING

    Error Details   COMPOSITE_ROUTING: Exception(s) were found for route(s): Route
    0: org.mule.runtime.core.api.retry.policy.RetryPolicyExhaustedException: ''until-successful''
    retries exhausted

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic duplicate issue'
  sentences:
  - 'INC0053356 Dear Team


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

    Transaction ID  05e66921-01b5-11f0-97d8-8624babe0d48

    Error Summary   500 CONNECTION_TIMEOUT

    Error Details   Could not establish SFTP connection with host: ''10.86.48.62''
    at port: ''22'' - timeout: socket is not established

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic scheduler ran and encountered a connection timeout but
    no data from SEGLINK'
  - "INC0051801 Rev007779 - the contract file doesn't have watermark after approved.\
    \ Kindly check and fix issue. Issue: No Watermark stamping in eReview document\
    \ 'Rev007779 ' final approval\r\nAnalysis: Issue not reproducible in Local/UAT\
    \ environment\r\nResolution: Resolved by SR ticket no. RITM0051289."
  - 'INC0053812 Dear Team


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

    Transaction ID  a18fc290-0593-11f0-a4be-3a952a31740d

    Error Summary   500 COMPOSITE_ROUTING

    Error Details   COMPOSITE_ROUTING: Exception(s) were found for route(s): Route
    0: org.mule.runtime.core.api.retry.policy.RetryPolicyExhaustedException: ''until-successful''
    retries exhausted

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic duplicate issue'
- source_sentence: "INC0017922 \"On 8/4/2024, Sasi san helped me upload a list of\
    \ assets on Genesis, but later we can not change any info detail of the assets\
    \ created by Sasisan on this day. Please check this example. Although there is\
    \ no EW, the error can not edit because this asset is linked to an active warranty\r\
    \nSerial: 67820350558\t\r\nModel: CU-XPU9XKH-8\r\n\" I checked with the Asset\
    \  67820350558 has no warranty registration records and No EW records, but still\
    \ the asset is in warranty."
  sentences:
  - INC0038847 eDO-M1-24002070 approved by Export Team Head but routing level in document
    still shows Export Team Head and document is not routed to next approver The issue
    was invalid context. I have to resubmit the document for approval.
  - 'INC0051436 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Flow Direction  inbound

    Subsidiary      PAPVN-TL2

    Source System   INVOICE

    End System      sap

    File Name       NA

    Storage Path

    API Name        sgst-fi-invoice-papi

    Error Source    INVOICE

    Transaction ID  bb011570-f284-11ef-ad6e-4efcac8a9e54

    Error Summary   HTTP GET on resource ''https://api.einvoice.fpt.com.vn:443/search-invoice''
    failed: Connect timeout.

    Error Details   HTTP GET on resource ''https://api.einvoice.fpt.com.vn:443/search-invoice''
    failed: Connect timeout.

    Comments        Unable to retrive file from Backup location


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Success on next run'
  - 'INC0054451 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Flow Direction  outbound

    Subsidiary      PAPVN-TL1

    Source System   SAP

    End System      einvoice

    File Name       0000000000057188

    Storage Path    /inbound/SAP/eInvoice/0000000000057188_To_einvoice.json

    API Name        sgst-fi-invoice-papi

    Error Source    INVOICE

    Transaction ID  09a169e0-09fb-11f0-ad2d-123bd73a1db4

    Error Summary   500 <html> <head> <meta name="viewport" content="width=device-width,
    initial-scale=1"> <style type="text/css"> /*! * Bootstrap v3.3.5 (http://getbootstrap.com)
    * Copyright 2011-2015 Twitter, Inc. * Licensed under MIT (https://github.com/twbs/bootstrap/blob/master/LICENSE)
    */ /*! normalize.css v3.0.3 | MIT License | github.com/necolas/normalize.css */
    html { font-family: sans-serif; -ms-text-size-adjust: 100%; -webkit-text-size-adjust:
    100%; } body { margin: 0; } h1 { font-size: 1.7em; font-weight: 400; line-height:
    1.3; margin: 0.68em 0; } * { -webkit-box-sizing: border-box; -moz-box-sizing:
    border-box; box-sizing: border-box; } *:before, *:after { -webkit-box-sizing:
    border-box; -moz-box-sizing: border-box; box-sizing: border-box; } html { -webkit-tap-highlight-color:
    rgba(0, 0, 0, 0); } body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    line-height: 1.66666667; font-size: 13px; color: #333333; background-color: #ffffff;
    margin: 2em 1em; } p { margin: 0 0 10px; font-size: 13px; } .alert.alert-info
    { padding: 15px; margin-bottom: 20px; border: 1px solid transparent; background-color:
    #f5f5f5; border-color: #8b8d8f; color: #363636; margin-top: 30px; } .alert p {
    padding-left: 35px; } a { color: #0088ce; } ul { position: relative; padding-left:
    51px; } p.info { position: relative; font-size: 15px; margin-bottom: 10px; } p.info:before,
    p.info:after { content: ""; position: absolute; top: 9%; left: 0; } p.info:before
    { content: "i"; left: 3px; width: 20px; height: 20px; font-family: serif; font-size:
    15px; font-weight: bold; line-height: 21px; text-align: center; color: #fff; background:
    #4d5258; border-radius: 16px; } @media (min-width: 768px) { body { margin: 4em
    3em; } h1 { font-size: 2.15em;} } </style> </head> <body> <div> <h1>Application
    is not available</h1> <p>The application is currently not serving requests at
    this endpoint. It may not have been started or is still starting.</p> <div class="alert
    alert-info"> <p class="info"> Possible reasons you are seeing this page: </p>
    <ul> <li> <strong>The host doesn''t exist.</strong> Make sure the hostname was
    typed correctly and that a route matching this hostname exists. </li> <li> <strong>The
    host exists, but doesn''t have a matching path.</strong> Check if the URL path
    was typed correctly and that the route was created using the desired path. </li>
    <li> <strong>Route and path matches, but all pods are down.</strong> Make sure
    that the resources exposed by this route (pods, services, deployment configs,
    etc) have at least one pod running. </li> </ul> </div> </div> </body> </html>

    Error Details   <html> <head> <meta name="viewport" content="width=device-width,
    initial-scale=1"> <style type="text/css"> /*! * Bootstrap v3.3.5 (http://getbootstrap.com)
    * Copyright 2011-2015 Twitter, Inc. * Licensed under MIT (https://github.com/twbs/bootstrap/blob/master/LICENSE)
    */ /*! normalize.css v3.0.3 | MIT License | github.com/necolas/normalize.css */
    html { font-family: sans-serif; -ms-text-size-adjust: 100%; -webkit-text-size-adjust:
    100%; } body { margin: 0; } h1 { font-size: 1.7em; font-weight: 400; line-height:
    1.3; margin: 0.68em 0; } * { -webkit-box-sizing: border-box; -moz-box-sizing:
    border-box; box-sizing: border-box; } *:before, *:after { -webkit-box-sizing:
    border-box; -moz-box-sizing: border-box; box-sizing: border-box; } html { -webkit-tap-highlight-color:
    rgba(0, 0, 0, 0); } body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    line-height: 1.66666667; font-size: 13px; color: #333333; background-color: #ffffff;
    margin: 2em 1em; } p { margin: 0 0 10px; font-size: 13px; } .alert.alert-info
    { padding: 15px; margin-bottom: 20px; border: 1px solid transparent; background-color:
    #f5f5f5; border-color: #8b8d8f; color: #363636; margin-top: 30px; } .alert p {
    padding-left: 35px; } a { color: #0088ce; } ul { position: relative; padding-left:
    51px; } p.info { position: relative; font-size: 15px; margin-bottom: 10px; } p.info:before,
    p.info:after { content: ""; position: absolute; top: 9%; left: 0; } p.info:before
    { content: "i"; left: 3px; width: 20px; height: 20px; font-family: serif; font-size:
    15px; font-weight: bold; line-height: 21px; text-align: center; color: #fff; background:
    #4d5258; border-radius: 16px; } @media (min-width: 768px) { body { margin: 4em
    3em; } h1 { font-size: 2.15em;} } </style> </head> <body> <div> <h1>Application
    is not available</h1> <p>The application is currently not serving requests at
    this endpoint. It may not have been started or is still starting.</p> <div class="alert
    alert-info"> <p class="info"> Possible reasons you are seeing this page: </p>
    <ul> <li> <strong>The host doesn''t exist.</strong> Make sure the hostname was
    typed correctly and that a route matching this hostname exists. </li> <li> <strong>The
    host exists, but doesn''t have a matching path.</strong> Check if the URL path
    was typed correctly and that the route was created using the desired path. </li>
    <li> <strong>Route and path matches, but all pods are down.</strong> Make sure
    that the resources exposed by this route (pods, services, deployment configs,
    etc) have at least one pod running. </li> </ul> </div> </div> </body> </html>

    Comments        Unable to retrive file from Backup location


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Reprocessed data.'
- source_sentence: INC0042827 Pls check PSV-SO-2410014343 CS LABOUR no DO The SO was
    successfully interfaced with SAP however the line item was not processed, we requested
    SAP team to re-process the confirmation and DO for the line item.
  sentences:
  - "INC0025126 Kindly Please help and Check User Cannot Approval 2 Claim Bellow :\r\
    \n1.claim-000025161\r\n2.claim-000025297\r\nWhen user (Ngadiyanto san) GDN HQ\
    \  try to approve From Partner Portal, appear Error  : \r\nInsufficient Access\
    \ Rights On Cross - Re ference ID,\r\nPlease see ServiceNow Attachment Closed\
    \ by Caller"
  - 'INC0044253 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Flow Direction  outbound

    Subsidiary      PAPVN-TL1

    Source System   SAP

    End System      einvoice

    File Name       0000000000045494

    Storage Path    /inbound/SAP/eInvoice/0000000000045494_To_einvoice.json

    API Name        sgst-fi-invoice-papi

    Error Source    INVOICE

    Transaction ID  90878250-b212-11ef-9d1e-b2c37322e2c2

    Error Summary   500 FPT''s response: Invoice number already existed

    Error Details   FPT''s response: Invoice number already existed

    Comments        Unable to retrive file from Backup location


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic The Invoice was sent twice hence the error'
  - INC0069108 delete al11 files email sent to you for your reference
- source_sentence: 'INC0054071 Dear Team


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

    Transaction ID  7625c5a0-07b4-11f0-a511-6a58f71b67be

    Error Summary   500 COMPOSITE_ROUTING

    Error Details   COMPOSITE_ROUTING: Exception(s) were found for route(s): Route
    0: org.mule.runtime.core.api.retry.policy.RetryPolicyExhaustedException: ''until-successful''
    retries exhausted

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic duplicate issue'
  sentences:
  - "INC0043099 Duplicate posting to SAP for eTravel: PCOA201359 but yet in the eTravel\
    \ application, it’s still shown as “Pending for claim“and Post flag as “scheduled“.\
    \ Why system allow duplicate posting to SAP?   \r\nPlease check. Dear Joyce san,\r\
    \nGood day!\r\n\r\nReg. incident ticket no. ‘INC0043099’.\r\nA gentle reminder(3)\
    \ for the below request.\r\n\r\nFor the reported document (PCOA201359) inquiry,\
    \ as update has been shared by SAP team earlier, an SR ticket is required to update\
    \ the ‘PostFlag’ & ‘Status’ fields in eWorkplace. So, kindly help to raise & share\
    \ the SR ticket no. for patching.\r\n\r\nAs no further updates received, from\
    \ business end after 3 reminder mails, we are proceeding to reslove/close this\
    \ incident ticket as per process.\r\nPlease let us know if any assistance required\
    \ further from AMS team.\r\n\r\nThanks for your understanding.\r\n\r\nThanks,\r\
    \nKannan B"
  - INC0051984 The issue is amount AP that displayed in FBV0 is different while it
    displayed in Y0NFI_0122. The correct one is in Y0NFI_0122, also after posted the
    journal was the same as Y0NFI_0122 in FB03. But due to different in FBV0, we have
    trouble while check before journal posting. Therefor, please help to solve this
    issue. Due to March is end of fiscal year, we would like to this issue solved
    within this week. Thank you for your support Explained to user that this is standard
    logic from SAP as per attached email
  - "INC0020015 when many WOs are created in the same time by Call Agents, system\
    \  frequently shows  \"duplicate value found\". It's takes long time to wait and\
    \ cant save. Its affects seriously to Call center's work performance. Kindly help\
    \ to fix in urgent From: Le Thi Thu Huyen <thuhuyen.le@vn.panasonic.com> \r\n\
    Sent: Thursday, June 6, 2024 2:32 PM\r\nTo: Nguyen Trung Thuc <trungthuc.nguyen@vn.panasonic.com>;\
    \ Mohammad Arif <mohammad.arif@sg.panasonic.com>\r\nCc: Dinesh Gatla <dinesh.gatla@sg.panasonic.com>\r\
    \nSubject: RE: Incident INC0020015 has been assigned to group CAPG L2 CRM\r\n\r\
    \nHi Team\r\n                Thank you for quick support. This below issue has\
    \ been resolved completely. Pls close the ticket.\r\n\r\nRegards,\r\nThu Huyen\r\
    \n\r\n\r\nFrom: Nguyen Trung Thuc <trungthuc.nguyen@vn.panasonic.com> \r\nSent:\
    \ Thursday, June 6, 2024 3:20 PM\r\nTo: Mohammad Arif <mohammad.arif@sg.panasonic.com>;\
    \ Le Thi Thu Huyen <thuhuyen.le@vn.panasonic.com>\r\nCc: Dinesh Gatla <dinesh.gatla@sg.panasonic.com>\r\
    \nSubject: RE: Incident INC0020015 has been assigned to group CAPG L2 CRM\r\n\r\
    \nHi @Le Thi Thu Huyen san,\r\nPlease help us check if this issue has been resolved\
    \ or not?\r\n\r\nBest regards,\r\n---------------------------------------------------------\r\
    \nNguyen Trung Thuc (Mr.)\r\nCS Planning Executive | Customer Service Department\r\
    \nPanasonic Sales Vietnam (PSV)"
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
    "INC0054071 Dear Team\n\nError occurred while processing the EDI transaction\n\nInterface       PANA.CART-PRD.MARKETING.Q\nSubsidiary      PSV\nAPI Name        pana-sf-mc-sapi\nFlow Direction  inbound\nSource System   PAPI\nEnd System      Salesforce Marketing Cloud\nFile Name       NA\nStorage Path    No Attachment\nError Source    Mulesoft\nTransaction ID  7625c5a0-07b4-11f0-a511-6a58f71b67be\nError Summary   500 COMPOSITE_ROUTING\nError Details   COMPOSITE_ROUTING: Exception(s) were found for route(s): Route 0: org.mule.runtime.core.api.retry.policy.RetryPolicyExhaustedException: 'until-successful' retries exhausted\nComments\n\nNote: This is an automated mail, please do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic duplicate issue",
    'INC0043099 Duplicate posting to SAP for eTravel: PCOA201359 but yet in the eTravel application, it’s still shown as “Pending for claim“and Post flag as “scheduled“. Why system allow duplicate posting to SAP?   \r\nPlease check. Dear Joyce san,\r\nGood day!\r\n\r\nReg. incident ticket no. ‘INC0043099’.\r\nA gentle reminder(3) for the below request.\r\n\r\nFor the reported document (PCOA201359) inquiry, as update has been shared by SAP team earlier, an SR ticket is required to update the ‘PostFlag’ & ‘Status’ fields in eWorkplace. So, kindly help to raise & share the SR ticket no. for patching.\r\n\r\nAs no further updates received, from business end after 3 reminder mails, we are proceeding to reslove/close this incident ticket as per process.\r\nPlease let us know if any assistance required further from AMS team.\r\n\r\nThanks for your understanding.\r\n\r\nThanks,\r\nKannan B',
    'INC0051984 The issue is amount AP that displayed in FBV0 is different while it displayed in Y0NFI_0122. The correct one is in Y0NFI_0122, also after posted the journal was the same as Y0NFI_0122 in FB03. But due to different in FBV0, we have trouble while check before journal posting. Therefor, please help to solve this issue. Due to March is end of fiscal year, we would like to this issue solved within this week. Thank you for your support Explained to user that this is standard logic from SAP as per attached email',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9737, 0.9816],
#         [0.9737, 1.0000, 0.9854],
#         [0.9816, 0.9854, 1.0000]])
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

* Size: 2,500 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                           | sentence_1                                                                           |
  |:--------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type    | string                                                                               | string                                                                               |
  | details | <ul><li>min: 12 tokens</li><li>mean: 137.73 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 12 tokens</li><li>mean: 142.86 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>INC0038248 All SO in List Excel is appear In SAP System & Partner Portal, But in SFDC is not appear (Cannot found when we searching), For detail see Excel Attachment. Issue: ISSUE - SO Number Cannot Read (Appear) in SALES FORCES  <br>Action plan taken by AMS: I have given the access to all the list of given so's . For the temporary fix we have raised problem ticket-PRB0040110, Team is working on it. No actions to be required from sfdc side. Hence closing the ticket.</code>                                                                                                                                                                                                                 | <code>INC0041974 Dear Team<br><br>Error occurred while processing the EDI transaction<br><br>Interface<br>Subsidiary      No Subsidiary<br>API Name        pana-pagitp-mgmt-eapi<br>Flow Direction  Inbound<br>Source System   PAGITP<br>End System      SAP S4Hana<br>File Name       No File Name<br>Storage Path    No Attachment<br>Error Source    Mulesoft<br>Transaction ID  a6f3ff16-e099-47b3-b157-490d311f4a0c<br>Error Summary   500 SOURCE_RESPONSE_SEND<br>Error Details   Client connection was closed<br>Comments<br><br>Note: This is an automated mail, please do not reply.<br><br>Thanks and Regards,<br><br>APAC Support, Panasonic 1. Checked the log in Cloudwatch.  <br>2. Checked the file in common storage.  <br>3. Verify the data if processed in MBP.  <br>Checked the FPL Invoice ECS4K1881006 is processed succesfully in MBP</code> |
  | <code>INC0063158 Idoc error message inv. IJP250609LY issue:  <br>Idoc failed to process the FPL document   <br>  <br>Category:  <br>SAP MM  <br>  <br>Failure point:  <br>Idoc failed to process the FPL document   <br>  <br>Root cause:  <br>As per analysis, even though system processed all the inbound delivery documents still it is showing the error Inbound delivery already exists. We suspect due to many number of inbound delivery documents issue causing. Monitoring in progress.   <br>Fix:  <br>NA  <br>Preventive Action:  <br>NA</code>                                                                                                                                                                      | <code>INC0036260 Please help to check why this planned order failed to convert? I suspect this issue because no routing found. but after we upload routing, it have an error. User has taking care of the production version with  correct routing group and counter.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
  | <code>INC0069321 Dear Team<br><br>Error occurred while processing the EDI transaction<br><br>Interface       GITP<br>Subsidiary      PIDSAP<br>API Name        inds-global-if-mgmt-papi<br>Flow Direction  Inbound<br>Source System   IBMMQ<br>End System      SAP<br>File Name       NA<br>Storage Path    No Attachment<br>Error Source    Mulesoft<br>Transaction ID  ID:363134323538393234303030303030303030303020202020<br>Error Summary   502 BAD_GATEWAY<br>Error Details   ***********443/api/v1/sender-receiver-code-reference-data' failed: bad gateway (502).<br>Comments<br><br>Note: This is an automated mail, please do not reply.<br><br>Thanks and Regards,<br><br>APAC Support, Panasonic</code> | <code>INC0014788 PACMY-SO-2404003526 ticket raised to wrong user.  <br>upon checkng SO already blocked and DO created</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
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

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 16
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
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
- `num_train_epochs`: 16
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
| Epoch   | Step | Training Loss |
|:-------:|:----:|:-------------:|
| 6.3291  | 500  | 3.99          |
| 12.6582 | 1000 | 3.4408        |


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