---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:5000
- loss:MatryoshkaLoss
- loss:MultipleNegativesRankingLoss
base_model: Qwen/Qwen3-Embedding-0.6B
widget:
- source_sentence: "INC0063589 And also help to cancel Invoice PMJI-2507-00392 before\
    \ 6pm today 4/7/2025 because else this will be posted to SAP which is having wrong\
    \ amount. Urgently seek assistance. Issue:Request to change the WO Charge Group\
    \ from Massage Chair to Ceiling Fan and cancel the invoice\r\nCategory: Internal\
    \ Data Update.\r\nRoot cause: User Requested to change the WO Charge Group from\
    \ Massage Chair to Ceiling Fan and cancel the invoice\r\nFix: Udpated the WO charge\
    \ group and cancelled the invoice.\r\nOutage: NA\r\nPreventive Action: NA\r\n\
    Outage: NA \r\nCR/Story: NA \r\nKnowledge Object: NA\r\nProblem Ticket: NA\r\n\
    Closure Evidence Attached (Y/N): Y"
  sentences:
  - INC0016107 Please cancel Extended warranty sales (EWS) PSV-EWS-2405000005 in SFDC
    PSV-EWS-2405000005 is cancelled
  - INC0060781 NOT EXIST IN T008 (CHECK ENTRY" Requested business to create SR ticket
    to create a rule for document change - Tcode OB32
  - INC0057770 Pls active Warranty registration W-5024636
- source_sentence: "INC0034520 Pls help us to fix the total amount of 42 WOs in the\
    \ attached list\r\nThanks! Issue: Wrong total amount\r\nAction Plan taken by AMS:\
    \ I have updated the total claim amount to the list of given all work orders.\
    \ No actions to be required from sfdc side. Hence closing the ticket."
  sentences:
  - "INC0041002 Dear AMS team,\r\nPAPVN (TL1) found the quantity of Target, Confirmed,\
    \ Delivered not matched at order no. 100835762, 100844851.\r\nPlease help to investigate\
    \ the issue.\r\nBrgs,\r\nChung resolved using program suggestion from SAP Global\
    \ to clear the inconsistency between the material document and delivery quantity\
    \ in COOIS"
  - 'INC0047605 Dear Team


    Error occurred while processing the EDI transaction


    Interface       PO-ACK

    Subsidiary      NA

    API Name        inds-global-if-mgmt-papi

    Flow Direction  Inbound

    Source System   SEGlink

    End System      SAP

    File Name       APRK14PCTL20250114150252931

    Storage Path    No Attachment

    Error Source    Mulesoft

    Transaction ID  34871240-d23d-11ef-ac26-9a7cb066868a

    Error Summary   500 CONNECTIVITY

    Error Details   ***********443/api/v1/mm/acknowledgment'' failed: Remotely closed.

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic A connection issue has been occurred when connecting to
    the S4Hana SAPI. File has been reprocessed and the verified that the acknowledgement
    has reflected in SAP'
  - "INC0034351 Kindly assist this case job in warranty but appear amount\r\n5000013208\t\
    WALK IN CUSTOMER (MC)\tPCMJI240801038\t29.08.2024\t9237101148\tY2\tMYR\t44.00\t\
    REQUEST DETAILS Hi MUHAMMAD FARIS BAHARUDDIN,\r\n\r\nAs per your confirmation\
    \ over the email, issue has been settled. Therefore, we are closing this ticket.\
    \ Please feel free to reopen it if you need further assistance."
- source_sentence: 'INC0068499 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Interface       GITP

    Subsidiary      PIDSAP

    API Name        inds-global-if-mgmt-papi

    Flow Direction  Outbound

    Source System   SAP

    End System      IBMMQ

    File Name       0000000003632508_6a389c10-821e-11f0-9935-3a6f305ca005.xml

    Storage Path    /INDS/prod/outbound/nocompany/ordrsp/0000000003632508_6a389c10-821e-11f0-9935-3a6f305ca005.xml

    Error Source    Mulesoft

    Transaction ID  6a389c10-821e-11f0-9935-3a6f305ca005

    Error Summary   500 CONNECTIVITY

    Error Details   ***********443/api/v1/sd/order/confirmation'' failed: Remotely
    closed.

    Comments        Unable to find 0000000003632508_6a389c10-821e-11f0-9935-3a6f305ca005.xml
    from Backup location


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic order response verified in gitp'
  sentences:
  - "INC0038305 \"Model: CS-YZ9AKH-8\r\nSerial: E212104065\r\nWhile linking warranty\
    \ data to EW side, this below error occurs, pls help us check & fix\r\nServer\
    \ Error : 500 : {\r\n\"\"errorCode\"\": 500,\r\n\"\"errorMessage\"\": \"\"HTTP\
    \ POST on resource 'https://smswarranty.vn.panasonic.com:9900/api/v1/sfdc-warranty-register'\
    \ failed: internal server error (500).\"\",\r\n\"\"correlationID\"\": \"\"e403a227-84a8-4af1-90d2-6b20c19c8e0c\"\
    \",\r\n\"\"timestamp\"\": \"\"2024-10-01-03-26-54\"\"\r\n}\" update database on\
    \ eW system"
  - 'INC0055624 Dear Team


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

    Transaction ID  5f48d48f-0abc-49d6-b2ce-9ad6c972e7d3

    Error Summary   500 SOURCE_RESPONSE_SEND

    Error Details   Client connection was closed

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic No action needed. Data processed successfully to SAP.'
  - "INC0032757 Spare part was ordered using PPH-WO-2408001172.\r\nWO line item created\
    \ = 2APRIKPM3520USB-S\r\nSales Order = PPH-SO-2408004036\r\nOrder products status\
    \ = part requested \r\n\r\nI cant cancel the part.\r\n\r\nPlease cancel the WOLI\
    \ and SO.\r\nWe will try to order the part again. Hi Rommel,\r\n\r\nWe assumed\
    \ that issue has been resolved, hence closing this ticket. Please reopen if you\
    \ want to further assistance on this."
- source_sentence: "INC0062653 Dear Team\n\nError occurred while processing the EDI\
    \ transaction\n\nInterface\nSubsidiary      No Subsidiary\nAPI Name        pana-pagitp-mgmt-eapi\n\
    Flow Direction  Inbound\nSource System   PAGITP\nEnd System      SAP S4Hana\n\
    File Name       No File Name\nStorage Path    No Attachment\nError Source    Mulesoft\n\
    Transaction ID  8b4f674f-6691-402a-914d-249a5a0bc509\nError Summary   500 SOURCE_RESPONSE_SEND\n\
    Error Details   Client connection was closed\nComments\n\nNote: This is an automated\
    \ mail, please do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic\
    \ 1. Checked the log in Cloudwatch.\r\n2. Checked the file in common storage.\r\
    \n3. Have verified that the data was processed in MBP."
  sentences:
  - INC0045921 I can't cancel PSV-WO-2408006938, pls help Cancel PSV-WO-2408006938
  - 'INC0059998 Dear Team


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

    Transaction ID  a5c0fb98-7a1a-4d58-8cec-4da475d82d13

    Error Summary   400 You have passed incorrect values. Please make sure to input
    the correct System ID and Client Number. For assistance kindly email apacmulesupport@sg.panasonic.com

    Error Details   You have passed incorrect values. Please make sure to input the
    correct System ID and Client Number. For assistance kindly email apacmulesupport@sg.panasonic.com

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic user provided invalid system id and client number combination'
  - 'INC0057821 Dear Team


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

    Transaction ID  1ae5da50-2ecd-11f0-ae4d-fa5087e70212

    Error Summary   500 CONNECTIVITY

    Error Details   Could not establish SFTP connection with host: ''10.86.48.62''
    at port: ''22'' - Session.connect: java.net.SocketTimeoutException: Read timed
    out

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Scheduler runs every 2 min. No action needed.'
- source_sentence: "INC0067467 Hi team, please refer to analysis from Andy below:\r\
    \n\r\nI ran a quick check on the difference between SAP and SFDC and while some\
    \ of the gaps have been fixed, we now have a new problem in some very old projects.\r\
    \n\r\nThe following projects have a lot of costs added in SFDC but almost nothing\
    \ in SAP as they have already been complete. There are obvious changes in SFDC\
    \ between 6th August (the time I ran the last report) and just now. You can see\
    \ in my table below. They are just the stand-out projects, I believe this affected\
    \ other projects as well just in a smaller scale.\r\n\r\nCan you please have a\
    \ look and make sure that the fix of the previous issue INC0066706  does not lead\
    \ to a new problem?\r\n\r\nThis is very critical as it is affecting business activities.\
    \ Please look into it with urgency.\r\n\r\nThanks Issue:\r\nExtra cost posting\
    \ variance in SFDC vs SAP\r\n\r\nCategory:\r\nCRM Integration\r\n\r\nFailure point:\r\
    \nMismatch in cost data between systems\r\n\r\nRoot cause:\r\nOld completed projects\
    \ in SAP received new cost entries in SFDC due to prior fix\r\n\r\nFix:\r\nReversed\
    \ additional cost entries in SFDC on 18th August; confirmed by stakeholders\r\n\
    \r\nPreventive Action:\r\nManual data rollback and coordination with COE and architecture\
    \ teams to identify gaps and streamline integration\r\n\r\nOutage:\r\nNA"
  sentences:
  - 'INC0044011 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Interface       GITP

    Subsidiary      PIDSTH

    API Name        inds-global-if-mgmt-papi

    Flow Direction  Outbound

    Source System   SAP

    End System      PAGITP

    File Name       0000000001748861_5b36bca0-b079-11ef-98df-9601f4f6b381.xml

    Storage Path    /INDS/prod/outbound/nocompany/purchase-order/0000000001748861_5b36bca0-b079-11ef-98df-9601f4f6b381.xml

    Error Source    PA-GITP

    Transaction ID  5b36bca0-b079-11ef-98df-9601f4f6b381

    Error Summary   500 INTERNAL_SERVER_ERROR

    Error Details   ***********443/api/v1/purchase-order'' failed: internal server
    error (500).

    Comments        Unable to find 0000000001748861_5b36bca0-b079-11ef-98df-9601f4f6b381.xml
    from Backup location


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Temporary connection problem occurred at the server in
    Japan. Reprocess has been  completed'
  - "INC0029876 Kindly Please help for this issue,\r\nAccount nana.poniman@id.panasonic.com\
    \  is cannot Verify Process for 2 WO (Invoiced) :\r\n1).PGI-WO-2406007375 / PGIJI-2406-06279\r\
    \n2).PGI-WO-2407008555 / PGIJI-2407-07179\r\nWhen nana.poniman@id.panasonic.com\
    \   try to Submitted appear error  Cannot Update After Work Order is Closed or\
    \ Job Claim Is Approved \r\nPlease fix this issue Issue: ISSUE - Cannot Verify\
    \ Claim \r\nAs I checked in the partner portal , the below claim id's already\
    \ approved.\r\n1).PGI-WO-2406007375 / PGIJI-2406-06279 - Claim-000025425\r\n2).PGI-WO-2407008555\
    \ / PGIJI-2407-07179 - Claim-000027425\r\nNo actions to be required from AMS sfdc\
    \ side. Hence closing the ticket."
  - "INC0056965 SALES SPEED DASHBOARD WITH WRONG DATA BY SALES STAFF AND BY DEALER\
    \ FOR THE MONTH APRIL  HAPPEN ON 28/4/2025. BEFORE THAT ITS GENERATED CORRECTLY.\
    \ Issue: SSD and KE30 report not matching \r\nCategory: SAP SD\r\nFailure point:\
    \ BP amount is not tally with Sales speed dashboard and KE30 ( Finance report\
    \ painter ) \r\nRoot cause: There were changes in salesman assigned for each dealer\
    \ hence need to re-run SSD update programs\r\nFix: Rerun SSD data collection program\
    \ and SSD Historical BP Update\r\nPreventive Action: Once there are changes in\
    \ salesman assigned to customers, must immediately notify AMS via SR ticket to\
    \ rerun SSD data collection program and SSD BP update\r\nOutage: NA\r\nCR/Story:\
    \ NA\r\nKnowledge Object: KB_AMS_SAP_OTC_NSC_PM SSRS BP Update\r\nProblem Ticket:\
    \ NA\r\nClosure Evidence Attached (Y/N): Y"
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
    "INC0067467 Hi team, please refer to analysis from Andy below:\r\n\r\nI ran a quick check on the difference between SAP and SFDC and while some of the gaps have been fixed, we now have a new problem in some very old projects.\r\n\r\nThe following projects have a lot of costs added in SFDC but almost nothing in SAP as they have already been complete. There are obvious changes in SFDC between 6th August (the time I ran the last report) and just now. You can see in my table below. They are just the stand-out projects, I believe this affected other projects as well just in a smaller scale.\r\n\r\nCan you please have a look and make sure that the fix of the previous issue INC0066706  does not lead to a new problem?\r\n\r\nThis is very critical as it is affecting business activities. Please look into it with urgency.\r\n\r\nThanks Issue:\r\nExtra cost posting variance in SFDC vs SAP\r\n\r\nCategory:\r\nCRM Integration\r\n\r\nFailure point:\r\nMismatch in cost data between systems\r\n\r\nRoot cause:\r\nOld completed projects in SAP received new cost entries in SFDC due to prior fix\r\n\r\nFix:\r\nReversed additional cost entries in SFDC on 18th August; confirmed by stakeholders\r\n\r\nPreventive Action:\r\nManual data rollback and coordination with COE and architecture teams to identify gaps and streamline integration\r\n\r\nOutage:\r\nNA",
]
documents = [
    'INC0056965 SALES SPEED DASHBOARD WITH WRONG DATA BY SALES STAFF AND BY DEALER FOR THE MONTH APRIL  HAPPEN ON 28/4/2025. BEFORE THAT ITS GENERATED CORRECTLY. Issue: SSD and KE30 report not matching \r\nCategory: SAP SD\r\nFailure point: BP amount is not tally with Sales speed dashboard and KE30 ( Finance report painter ) \r\nRoot cause: There were changes in salesman assigned for each dealer hence need to re-run SSD update programs\r\nFix: Rerun SSD data collection program and SSD Historical BP Update\r\nPreventive Action: Once there are changes in salesman assigned to customers, must immediately notify AMS via SR ticket to rerun SSD data collection program and SSD BP update\r\nOutage: NA\r\nCR/Story: NA\r\nKnowledge Object: KB_AMS_SAP_OTC_NSC_PM SSRS BP Update\r\nProblem Ticket: NA\r\nClosure Evidence Attached (Y/N): Y',
    "INC0029876 Kindly Please help for this issue,\r\nAccount nana.poniman@id.panasonic.com  is cannot Verify Process for 2 WO (Invoiced) :\r\n1).PGI-WO-2406007375 / PGIJI-2406-06279\r\n2).PGI-WO-2407008555 / PGIJI-2407-07179\r\nWhen nana.poniman@id.panasonic.com   try to Submitted appear error  Cannot Update After Work Order is Closed or Job Claim Is Approved \r\nPlease fix this issue Issue: ISSUE - Cannot Verify Claim \r\nAs I checked in the partner portal , the below claim id's already approved.\r\n1).PGI-WO-2406007375 / PGIJI-2406-06279 - Claim-000025425\r\n2).PGI-WO-2407008555 / PGIJI-2407-07179 - Claim-000027425\r\nNo actions to be required from AMS sfdc side. Hence closing the ticket.",
    "INC0044011 Dear Team\n\nError occurred while processing the EDI transaction. Please find the details below and attached is the file associated to the transaction.\n\nInterface       GITP\nSubsidiary      PIDSTH\nAPI Name        inds-global-if-mgmt-papi\nFlow Direction  Outbound\nSource System   SAP\nEnd System      PAGITP\nFile Name       0000000001748861_5b36bca0-b079-11ef-98df-9601f4f6b381.xml\nStorage Path    /INDS/prod/outbound/nocompany/purchase-order/0000000001748861_5b36bca0-b079-11ef-98df-9601f4f6b381.xml\nError Source    PA-GITP\nTransaction ID  5b36bca0-b079-11ef-98df-9601f4f6b381\nError Summary   500 INTERNAL_SERVER_ERROR\nError Details   ***********443/api/v1/purchase-order' failed: internal server error (500).\nComments        Unable to find 0000000001748861_5b36bca0-b079-11ef-98df-9601f4f6b381.xml from Backup location\n\nNote: This is an automated mail, please do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic Temporary connection problem occurred at the server in Japan. Reprocess has been  completed",
]
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# [1, 1024] [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[0.9957, 0.9940, 0.9934]])
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
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          768,
          512,
          256,
          128
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
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
| 1.5974 | 500  | 13.6058       |
| 1.9936 | 624  | -             |
| 2.0    | 626  | -             |
| 2.4920 | 780  | -             |
| 2.9904 | 936  | -             |
| 3.0    | 939  | -             |
| 3.1949 | 1000 | 12.8553       |
| 3.4888 | 1092 | -             |
| 3.9872 | 1248 | -             |
| 4.0    | 1252 | -             |
| 0.4984 | 156  | -             |
| 0.9968 | 312  | -             |
| 1.0    | 313  | -             |
| 1.4952 | 468  | -             |
| 1.5974 | 500  | 13.9763       |
| 1.9936 | 624  | -             |
| 2.0    | 626  | -             |
| 2.4920 | 780  | -             |
| 2.9904 | 936  | -             |
| 3.0    | 939  | -             |
| 3.1949 | 1000 | 13.825        |
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

#### MatryoshkaLoss
```bibtex
@misc{kusupati2024matryoshka,
    title={Matryoshka Representation Learning},
    author={Aditya Kusupati and Gantavya Bhatt and Aniket Rege and Matthew Wallingford and Aditya Sinha and Vivek Ramanujan and William Howard-Snyder and Kaifeng Chen and Sham Kakade and Prateek Jain and Ali Farhadi},
    year={2024},
    eprint={2205.13147},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
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