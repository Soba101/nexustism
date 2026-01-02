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
- source_sentence: INC0059938 Output YD13 missing from DO 255025855. Please check
    As the YD13 output type was already present in the DO, and the user has confirmed
    closure, no further action is required.
  sentences:
  - 'INC0059046 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Flow Direction  outbound

    Subsidiary      PAPVN-TL1

    Source System   SAP

    End System      SCM_NAVI

    File Name       SCM_NAVI_11P1_PRODUCTION_PLAN

    Storage Path

    API Name        sgst-pp-edi-eapi

    Error Source    Mulesoft

    Transaction ID  df512d19-3202-4c0e-81d1-ce2d529d3b73

    Error Summary   500 HTTP POST on resource ''https://sgst-hulft-sapi-qoq0kf.internal-hnygb7.sgp-s1.cloudhub.io:443/api/v1/production-plan''
    failed: Timeout exceeded.

    Error Details   HTTP POST on resource ''https://sgst-hulft-sapi-qoq0kf.internal-hnygb7.sgp-s1.cloudhub.io:443/api/v1/production-plan''
    failed: Timeout exceeded.

    Comments        Unable to retrive file from Backup location


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Transaction was successfully posted to HULFT STFP server
    but encountered during backup process.'
  - "INC0031186 Dear Support team,\r\nPlease help to check detail as attached file\
    \ and support us on the query.\r\nThank you for your support. Program Bug has\
    \ been fix, & changes are moved to production, Results are now showing correct"
  - "INC0022101 These are the paid order but doesn’t push to OMS, and we noticed the\
    \ last order pushed to SAP is Jun 6, 11:14 AM. \r\n\r\nCan you please check and\
    \ resolve this issue by today?\r\n\r\n9000007503\r\n9000007506\r\n9000007509\r\
    \n9000007512\r\n9000007515\r\n9000007518\r\n9000007521\r\n9000007524\r\n9000007527\r\
    \n9000007530\r\n9000007536\r\n9000007539\r\n9000007542\r\n9000007545\r\n9000007551\
    \ checked all Customer PO and created in SAP\r\nthere was no issues found in SAP\r\
    \n@[Youka Toh] which report are you checking in SAP why not able to see SAP Sales\
    \ orders?"
- source_sentence: 'INC0045591 Dear Team


    Error occurred while processing the EDI transaction


    Interface       avnet

    Subsidiary      PIDSMY

    API Name        ext-partners-order-mgmt-papi

    Flow Direction  Inbound

    Source System   SAP

    End System      NA

    File Name       NA

    Storage Path    No Attachment

    Error Source    Mulesoft

    Transaction ID  713096c1-bdfd-11ef-9243-fa55cdd6108b

    Error Summary   500 TIMEOUT

    Error Details   ***********443/api/v1/stock-code/AVNET%20OTHERS'' failed: Timeout
    exceeded.

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic There is an intermittent connection issue happened when
    processing the files. It was also found that the ext-partners=order-mgmt-papi
    was hitting 100% util during the processing which is also contributed to the error.
    Reprocess has been completed.'
  sentences:
  - INC0030259 Fixed cost not showing in CU-RU12AKD-31 Plant IAF1 '@[SANDEEP SAIN],
    As discussed over call, the material is procuring from outside, so it does not
    have fixed cost. Only inhouse manufacturing products can have fixed cost in CK13N.
  - 'INC0065693 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Interface       RIDM - GID EMailId

    Subsidiary      PAPAMY

    API Name        pana-hriq-mgmt-papi

    Flow Direction  inbound

    Source System   RIDM API

    End System      pana-global-hriq-sapi

    File Name       ridm-gid_emailid_ac85f800-6a0f-11f0-8535-9e579f404550.json

    Storage Path    /PAPAMY/ridm/gid-emailid/ridm-gid_emailid_ac85f800-6a0f-11f0-8535-9e579f404550.json

    Error Source    SuccessFactor-OData API

    Transaction ID  ac85f800-6a0f-11f0-8535-9e579f404550

    Error Summary   400 BAD_REQUEST

    Error Details   Error response from SuccessFactors API

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic raised to trung / jane'
  - INC0055075 When business re-submit the MY invoices for e-invoice validation, the
    cached error message will appear and stop the re-submission. User confirmed the
    issue is resolved and asked to close the incident
- source_sentence: "INC0018084 Please check why 78116CBM, 78116DBL & 78116DPC is missing\
    \ from the COI listing.  Approval will be:\r\nHilary Approver 1 and Sammi Tanaka\
    \ Approver 2\r\n\r\nPlease inform after correction to on behalf requestor and\
    \ jenny.phua@sg.panasonic.com Since we have provided the information about PTLAP\
    \ user inquiry on this request. Once changes moved into prod instance, we will\
    \ close this ticket \"RITM0018821\" and let you know the update. \r\nHence confirmed\
    \ with your end, we are proceeding to close this ticket \"INC0018084\""
  sentences:
  - 'INC0060676 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Flow Direction

    Subsidiary

    Source System   SAP

    End System      EPRO

    File Name       Y0GMM_ZAO0110_R41_ID20_20250611170009

    Storage Path

    API Name        sgst-audit-papi

    Error Source    PCS

    Transaction ID  4fe66c67-9b40-4fa2-9484-90a67f1d5ee9

    Error Summary   500 Exception was found writing to file ''/outbound/pmi/epro/purchase-orders/Y0GMM_ZAO0110_R41_ID20_20250611170009_4fe66c67-9b40-4fa2-9484-90a67f1d5ee9.txt''

    Error Details   Exception was found writing to file ''/outbound/pmi/epro/purchase-orders/Y0GMM_ZAO0110_R41_ID20_20250611170009_4fe66c67-9b40-4fa2-9484-90a67f1d5ee9.txt''

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Error writing to PCS backup server. No action needed since
    file was posted to GITP'
  - "INC0017879 Interface\t        employee-minimaster-ridm\r\nSubsidiary\tPAPAMY\r\
    \nAPI Name\t         pana-global-hriq-sapi\r\nFlow Direction\tinbound\r\nSource\
    \ System\tsuccessFactors\r\nEnd System\t       RIDM\r\nFile Name\t           \
    \    ridm-empminimaster_12918350-124e-11ef-ae57-ca65c4658f1c.json\r\nStorage Path\t\
    \       No Attachment\r\nError Source\t        RIDM\r\nTransaction ID\t2d37dec0-124e-11ef-91bd-8a161923f34c\r\
    \nError Summary\t400 BAD_REQUEST\r\nError Details\t        Error response from\
    \ RIDM API the third-party vendor has informed of the incident and will be resolve\
    \ on their end"
  - 'INC0059575 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Flow Direction  inbound

    Subsidiary      PAPVN-TL1

    Source System   INVOICE

    End System      sap

    File Name       NA

    Storage Path

    API Name        sgst-fi-invoice-papi

    Error Source    INVOICE

    Transaction ID  501c7380-3da7-11f0-b397-fe052cc11875

    Error Summary   <html> <head> <meta name="viewport" content="width=device-width,
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


    APAC Support, Panasonic Downstream FPT issue. Have notified PICs and acknowledged
    this issue, PICs will be checking with external vendor. No action from MuleSoft
    required.'
- source_sentence: 'INC0060445 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Flow Direction

    Subsidiary

    Source System   SAP

    End System      PBS

    File Name       VN11_MainlineInput

    Storage Path

    API Name        sgst-audit-papi

    Error Source    PCS

    Transaction ID  caa3d710-45a9-11f0-8b28-52ac189c79c9-1

    Error Summary   500 Could not establish SFTP connection with host: ''10.86.48.62''
    at port: ''22'' - Session.connect: java.net.SocketTimeoutException: Read timed
    out

    Error Details   Could not establish SFTP connection with host: ''10.86.48.62''
    at port: ''22'' - Session.connect: java.net.SocketTimeoutException: Read timed
    out

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic SFTP server issue.  No action required.'
  sentences:
  - 'INC0060202 Dear Team


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

    Transaction ID  f97f9807-aed5-4430-ad04-9fd54e9953d3

    Error Summary   400 Unable to connect to SAP.Enter Correct User Name.. Please
    contact Basis at piscap-basis-team@sg.panasonic.com.

    Error Details   Unable to connect to SAP.Enter Correct User Name.. Please contact
    Basis at piscap-basis-team@sg.panasonic.com.

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic user provided incorrect username'
  - "INC0031836 Dear Surya san, Bandi san and PISCAP SAP SD BRS, PISCAP SAP SD BRS\
    \ team,\r\n\r\nBased on my conversation with the PGI Finance team,\r\n\r\nWe found\
    \ 2 SAP Customer Code data for PT. DATASCRIP\r\n1.5000026094 ? is account for\
    \ System Solution PGI \r\n2.5000015660 ? is have account for access Genesis Partner\
    \ Portal  (Using by Vendor PGI) \r\n\r\nOur question is : \r\n1). Can SAP Customer\
    \ Code 5000015660 be replaced with SAP Customer Code 5000026094 (made the same)?\
    \ and what impact will this change have?\r\nBecause this is to prevent an AR Overdue\
    \ from purchasing Spare Parts (internal problems ==> DN at the end of the month),\
    \ then it has no impact on AR Trade.\r\n\r\n2). If Point 2 can be done, what is\
    \ the replacement process? Does it require Update By DMR?\r\n\r\nThanks and we\
    \ are waiting for the information closed upon users confirmation"
  - 'INC0060446 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Flow Direction

    Subsidiary

    Source System   SAP

    End System      PBS

    File Name       VN11_UrethaneNo1

    Storage Path

    API Name        sgst-audit-papi

    Error Source    PCS

    Transaction ID  caa3d710-45a9-11f0-8b28-52ac189c79c9-3

    Error Summary   500 Could not establish SFTP connection with host: ''10.86.48.62''
    at port: ''22'' - Error during login to mule2Ppana@10.86.48.62

    Error Details   Could not establish SFTP connection with host: ''10.86.48.62''
    at port: ''22'' - Error during login to mule2Ppana@10.86.48.62

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic SFTP server issue.  No action required.'
- source_sentence: INC0011410 Please change Extended warranty sales (EWS) from W-5236338
    to W-8260398 (expiry date 11/06/2028 after create EWS) Changed the warranty W-5236338
    this from the EWS to W-8260398 , Also created the EW for the warranty.
  sentences:
  - 'INC0056820 Dear Team


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

    Transaction ID  9a060ea6-dc00-42b5-8b4b-70b819e94c0e

    Error Summary   500 SOURCE_RESPONSE_SEND

    Error Details   Client connection was closed

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic No action needed. YEMM_ORDRSP iDoc posted successfully
    to SAP'
  - "INC0010971 B0DCCF000002 VMI stock quantity should be 0 but it showed 125000 in\
    \ API report. Please help to check the reason and how to correct it.\r\n\r\nThis\
    \ case is related to month-end closing, please give it priority. In order to correct\
    \ the quantity we need to apply for a Power ID and modify the VMI table ZCGAMM_T0010"
  - 'INC0045589 Dear Team


    Error occurred while processing the EDI transaction


    Interface       avnet

    Subsidiary      PIDSMY

    API Name        ext-partners-order-mgmt-papi

    Flow Direction  Inbound

    Source System   SAP

    End System      NA

    File Name       NA

    Storage Path    No Attachment

    Error Source    Mulesoft

    Transaction ID  713096c2-bdfd-11ef-9243-fa55cdd6108b

    Error Summary   500 TIMEOUT

    Error Details   ***********443/api/v1/file-reference-data/AVN214SX'' failed: Timeout
    exceeded.

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic There is an intermittent connection issue happened when
    processing the files. It was also found that the ext-partners=order-mgmt-papi
    was hitting 100% util during the processing which is also contributed to the error.
    Reprocess has been completed.'
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
    'INC0011410 Please change Extended warranty sales (EWS) from W-5236338 to W-8260398 (expiry date 11/06/2028 after create EWS) Changed the warranty W-5236338 this from the EWS to W-8260398 , Also created the EW for the warranty.',
    "INC0045589 Dear Team\n\nError occurred while processing the EDI transaction\n\nInterface       avnet\nSubsidiary      PIDSMY\nAPI Name        ext-partners-order-mgmt-papi\nFlow Direction  Inbound\nSource System   SAP\nEnd System      NA\nFile Name       NA\nStorage Path    No Attachment\nError Source    Mulesoft\nTransaction ID  713096c2-bdfd-11ef-9243-fa55cdd6108b\nError Summary   500 TIMEOUT\nError Details   ***********443/api/v1/file-reference-data/AVN214SX' failed: Timeout exceeded.\nComments\n\nNote: This is an automated mail, please do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic There is an intermittent connection issue happened when processing the files. It was also found that the ext-partners=order-mgmt-papi was hitting 100% util during the processing which is also contributed to the error. Reprocess has been completed.",
    'INC0010971 B0DCCF000002 VMI stock quantity should be 0 but it showed 125000 in API report. Please help to check the reason and how to correct it.\r\n\r\nThis case is related to month-end closing, please give it priority. In order to correct the quantity we need to apply for a Power ID and modify the VMI table ZCGAMM_T0010',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.3944, 0.5700],
#         [0.3944, 1.0000, 0.3542],
#         [0.5700, 0.3542, 1.0000]])
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
  |         | sentence_0                                                                          | sentence_1                                                                           | label                                                         |
  |:--------|:------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                              | string                                                                               | float                                                         |
  | details | <ul><li>min: 17 tokens</li><li>mean: 132.6 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 16 tokens</li><li>mean: 136.18 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.5</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>INC0030502 eTravel2795 cannot click Travel type even the document is RFI Dear Eunice san,  <br>Good evening!  <br>  <br>As discussed in teams, for this eTravel document 'eTravel2795', the travel type cannot be modified in Post stage, and this is as per design. So, we have to patch the data as discussed. Hence SR ticket (RITM0027625) has been raised, we are proceeding to resolve/close this incident ticket upon confirmation.  <br>  <br>Thanks,  <br>Kannan B</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | <code>INC0052748 request to cancel this SO#PPH-SO-2502001839,PPH-SO-2502000735 parts stuck as " Part requested" As request is fulfilled in INC0052670.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | <code>1.0</code> |
  | <code>INC0051853 SAP Transaction code Y0NSD_0046 Sales order report SR  is created for this request</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | <code>INC0043785 Dear Team<br><br>Error occurred while processing the EDI transaction<br><br>Interface       micron<br>Subsidiary      PIDSMY<br>API Name        ext-partners-order-mgmt-papi<br>Flow Direction  Inbound<br>Source System   SAP<br>End System      NA<br>File Name       NA<br>Storage Path    No Attachment<br>Error Source    Mulesoft<br>Transaction ID  701aefb0-ada4-11ef-a09a-0a9283444228<br>Error Summary   400 BAD_REQUEST<br>Error Details   Segment group repeated too many times past end of segment 38 in message 1 of interchange 362<br>Comments<br><br>Note: This is an automated mail, please do not reply.<br><br>Thanks and Regards,<br><br>APAC Support, Panasonic data issue - ORDCHG sent by Micron has more than 10 (max) QTY/DTM iterations (see attached email for details)  <br>Micron side will remove the DTM+48 in the ORDCHG to Panasonic</code> | <code>1.0</code> |
  | <code>INC0043349 Dear Team<br><br>Error occurred while processing the EDI transaction<br><br>Interface       internal-stocks<br>Subsidiary      PAU<br>API Name        pana-sdesk-ext-eapi<br>Flow Direction  outbound<br>Source System   PAU<br>End System      ZOHO<br>File Name       No File Name<br>Storage Path    No Attachment<br>Error Source    pana-sdesk-ext-eapi<br>Transaction ID  dcb44460-aaec-11ef-8b0c-02b4130d4440<br>Error Summary   500 SOURCE_RESPONSE_SEND<br>Error Details   SOURCE_RESPONSE_SEND: '/Zoho/Live/Outbound/status/remote/SPR_Status_20241121090803362047.csv' cannot be renamed because '/Zoho/Live/Outbound/status/done/SPR_Status_20241121090803362047.csv' already exists<br>Comments<br><br>Note: This is an automated mail, please do not reply.<br><br>Thanks and Regards,<br><br>APAC Support, Panasonic The setup for the file poller in outbound servicedesk is it does not do overwrite, thus, the issue encountered. As a workaround, the previous file has been deleted. As for the resolution, overwrite is now enabled.</code> | <code>INC0062039 Users found a scenario where one parent ID may have 2 factory IDs, however the 1 item will only fall under 1 factory ID.  However 2 records for for the item, one for each factory ID.    Kindly check. Query to PA PUBLIC was modified.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | <code>0.0</code> |
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
| Epoch   | Step | Training Loss |
|:-------:|:----:|:-------------:|
| 1.5974  | 500  | 0.2123        |
| 3.1949  | 1000 | 0.2072        |
| 4.7923  | 1500 | 0.2025        |
| 6.3898  | 2000 | 0.1995        |
| 7.9872  | 2500 | 0.197         |
| 9.5847  | 3000 | 0.1945        |
| 11.1821 | 3500 | 0.1934        |


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