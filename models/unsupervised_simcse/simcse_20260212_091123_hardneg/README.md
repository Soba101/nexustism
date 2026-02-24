---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:1984
- loss:CosineSimilarityLoss
widget:
- source_sentence: 'INC0068690 Dear Team


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

    Transaction ID  be6c073b-d490-4f4a-8c31-2e0834819909

    Error Summary   500 SOURCE_RESPONSE_SEND

    Error Details   Client connection was closed

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic YEMM_IPL_INVOICE01 iDoc posted successfully to SAP'
  sentences:
  - 'INC0045587 Dear Team


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

    Transaction ID  28ee5be1-bdfd-11ef-9243-fa55cdd6108b

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
  - 'INC0061076 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Flow Direction

    Subsidiary

    Source System   SAP

    End System      SALES_EDI

    File Name       PGI_INVOICE_ID20_20250616144444

    Storage Path

    API Name        sgst-audit-papi

    Error Source    PCS

    Transaction ID  fe99e9f4-83a4-4893-9c87-7122fb7825e9

    Error Summary   500 Exception was found writing to file ''/outbound/pmi/sales_edi/fpl/PGI_INVOICE_ID20_20250616144444_fe99e9f4-83a4-4893-9c87-7122fb7825e9.txt''

    Error Details   Exception was found writing to file ''/outbound/pmi/sales_edi/fpl/PGI_INVOICE_ID20_20250616144444_fe99e9f4-83a4-4893-9c87-7122fb7825e9.txt''

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Error writing to PCS backup server. No action needed since
    file was posted to GITP'
  - 'INC0059410 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Interface       GITP

    Subsidiary      PIDSAP

    API Name        inds-global-if-mgmt-papi

    Flow Direction  Outbound

    Source System   SAP

    End System      IBMMQ

    File Name       0000000002997372_ca95f750-3c88-11f0-8437-4a15ca644d9e.xml

    Storage Path    /INDS/prod/outbound/nocompany/ordrsp/0000000002997372_ca95f750-3c88-11f0-8437-4a15ca644d9e.xml

    Error Source    Mulesoft

    Transaction ID  ca95f750-3c88-11f0-8437-4a15ca644d9e

    Error Summary   500 CONNECTIVITY

    Error Details   ***********443/api/v1/sd/order/confirmation'' failed: Remotely
    closed.

    Comments        Unable to find 0000000002997372_ca95f750-3c88-11f0-8437-4a15ca644d9e.xml
    from Backup location


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Transaction was posted to GITP. No action required.'
- source_sentence: 'INC0049841 Dear Team


    Error occurred while processing the EDI transaction


    Interface       approved-claims

    Subsidiary      eworkplace

    API Name        pana-global-hriq-sapi

    Flow Direction  inbound

    Source System

    End System

    File Name       eWorkplace_eCLAIM_06022025.csv

    Storage Path    No Attachment

    Error Source    eWorkplace-FTP

    Transaction ID  751d37b0-e4b4-11ef-a618-4aeb6835e7b1

    Error Summary   500 RUNTIME_ERROR

    Error Details   Could not establish FTP connection with host: ''10.81.24.89''
    at port: ''21'' - Error code: 530 - User cannot log in

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Manually Reprocessed.'
  sentences:
  - "INC0046988 While executing the transaction ZRNMM_0006 user is facing dump issue\
    \ but from CAPG side we checked we don't have any dump issue.\r\nBelow attached\
    \ the User IDs need to check, why facing issue.\r\n\r\nUsed GID\r\n70C2106\r\n\
    70G8569\r\n70D8869 issue is not related to access but more with PC as user is\
    \ able to download data using local laptop. \r\n\r\n\r\nFrom: Vilurdena Munusamy\
    \ \r\nSent: Thursday, 9 January, 2025 6:43 PM\r\nTo: Pandurang Mane; PISCAP_SAP_Basis_Team;\
    \ 'Wahyu setiawan' <wahyu.setiawan@id.panasonic.com>\r\nCc: Gaddam Jonathan Jaypal\
    \ Raj; Ganesh Padala; PISCAP_SAP_MM\r\nSubject: RE: INC0046945 - [PTP] Dump error\
    \ when download data via tcode ZRNMM_0006\r\n\r\nHi @Wahyu setiawan and @Pandurang\
    \ Mane-san,\r\n\r\nAs confirmed in teams chat we will proceed to close the incident:\
    \ INC0046988"
  - "INC0040867 Please help to check, we found an error when costing some material.\
    \ \r\nfor example material CS-N18AKH-8-ASSY. error message I attached. please\
    \ do high priority due to month end clossing User having issue in ck11n . then\
    \ after that we have provided our analyses to the user with pp team and issue\
    \ is resolved and Emilien San provided closer in teams chat."
  - "INC0041453 User : Joann  \r\nBP 80910218\r\nChange :  05.07.2024 16:30:24 , \
    \ tax classification was change from 2 to 1 by 50SUPPORT04 \r\n\r\nPls kindly\
    \ advise under which SR and who requested this change.\r\n\r\nImpact : \r\nDue\
    \ to above change to 1, all the billing documents are affected . \r\nWe have already\
    \ raised another SR change back to 2 on 30.10.2024 due to eWork Customer master\
    \ was still keeping Tax Class 1. analyses provided to user and this mistake is\
    \ done by"
- source_sentence: INC0024164 cannot access ework Seems there was a network drop and
    server was restarted the user login is successful
  sentences:
  - "INC0051652 Please check why daily allowance are not interface for many eTravel\
    \ documents. Please do data extraction and check all affected records.  \r\nOnly\
    \ quoting few examples, PISCAP205529, PISCAP205384, PTLAP205535, PCMAP205545,\
    \ PPAP205610, MIG205540 Dear @Bee Yen Loh san,\r\nGood morning!!!\r\n\r\nPlease\
    \ find the below:\r\n\r\nIssue:\r\n•\tHRIQ posting failure in the eTravel application.\r\
    \nCause: \r\n•\tDue to recent Azure migration.\r\nAnalysis: \r\n•\tUpon verification,\
    \ the issue was identified to be caused by the following reasons:\r\no\tThe Azure\
    \ migration activity.\r\no\tThe incorrect selection of the ‘Staff Type’ (\"Agency\
    \ or Contract\") by the applicant in the eTravel document prevented the entire\
    \ posting (SAP & HRIQ) from being triggered from eWorkplace.\r\nSolution: \r\n\
    •\tWe have rectified the HRIQ posting configuration and re-posted the affected\
    \ documents based on the reasons mentioned above.\r\n\r\nCountermeasures:\r\n\
    •\tFor server-related migration activities, our solution team has verified all\
    \ the connecting systems (related to Posting) during the migration period.\r\n\
    •\tSince this migration activity was conducted in the first week of February,\
    \ we have monitored and shared the consolidated list of affected documents with\
    \ you, and they have been reprocessed in the HRIQ system.\r\n•\tApplicants must\
    \ select the correct 'Staff Type' while submitting the document. An incorrect\
    \ Staff Type value will lead to Posting issues in SAP, and a failure email notification\
    \ will be sent to all eTravel admins and the AMS team.\r\n\r\nCurrent Process:\r\
    \n•\tPosting to HRIQ and SAP systems are triggered at different times. If a document\
    \ faces posting failure, a notification will be sent to the business user in the\
    \ eTravel admin group and the AMS team.\r\n•\tUpon receiving a ticket from the\
    \ business regarding a posting failure, we will create a child incident ticket\
    \ for the SAP team, along with the failure posting log, to investigate the issue.\r\
    \n•\tAfter the SAP team rectifies the issue, if an SR ticket is required, we will\
    \ inform the business with the resolution notes to resolve the posting issue in\
    \ the reported eTravel documents.\r\n\r\nGiven that this is a rare scenario resulting\
    \ from an incorrect 'Staff Type' selection in the documents, we will take note\
    \ of it going forward and ensure extra monitoring until the affected document\
    \ is fully processed in the SAP & HRIQ system.\r\n\r\nWe appreciate your understanding\
    \ and cooperation!\r\n\r\nHence confirmed with your end, we are proceeding to\
    \ close the below ticket.\r\n\r\n•\tRITM0050009\r\n•\tINC0051652\r\n\r\n\r\n\r\
    \nRegards,\r\nManoj.BK\r\nIT Consultant\r\nPanasonic Information Systems Company\
    \ Asia Pacific (PISCAP) \r\nE-mail: manoj.bk@sg.panasonic.com"
  - "INC0042897 User: Josie/ Katrinna (EAI HRIQ staff)\r\n\r\neTR-S1-24000237 - HRIQ\
    \ posting to SAP error\r\nPls see error attached in email. Pls help to advise\
    \ how to resolve this posting error issue is resolved and document posted in service\
    \ now confirmed by service now they are not getting any issue. email attach in\
    \ attachment"
  - 'INC0059667 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Flow Direction

    Subsidiary

    Source System   SAP

    End System      EPRO

    File Name       Y0GMM_ZAO0110_R41_ID20_20250602090100

    Storage Path

    API Name        sgst-audit-papi

    Error Source    PCS

    Transaction ID  47728f62-7df7-4744-bb0e-dcfd04af9e72

    Error Summary   500 Exception was found writing to file ''/outbound/pmi/epro/purchase-orders/Y0GMM_ZAO0110_R41_ID20_20250602090100_47728f62-7df7-4744-bb0e-dcfd04af9e72.txt''

    Error Details   Exception was found writing to file ''/outbound/pmi/epro/purchase-orders/Y0GMM_ZAO0110_R41_ID20_20250602090100_47728f62-7df7-4744-bb0e-dcfd04af9e72.txt''

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Error writing to PCS backup server. No action needed since
    file was posted to GITP'
- source_sentence: 'INC0068587 Dear Team


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

    Transaction ID  9e3025cf-19cf-4f2e-8407-701434e2edde

    Error Summary   500 SOURCE_RESPONSE_SEND

    Error Details   Client connection was closed

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic YEMM_IPL_INVOICE01 iDoc posted successfully to SAP'
  sentences:
  - 'INC0059376 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Flow Direction

    Subsidiary

    Source System   SAP

    End System      PBS

    File Name       VN11_UrethaneNo6

    Storage Path

    API Name        sgst-audit-papi

    Error Source    PCS

    Transaction ID  f2fc0f10-3c54-11f0-8b28-52ac189c79c9-11

    Error Summary   500 Error encountered

    Error Details   Error encountered

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Error writing to PCS backup server. No action needed since
    file was posted to SAP and PBS'
  - 'INC0059316 Dear Team


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

    Transaction ID  120458b5-1960-4396-b61d-b29252cdf5fe

    Error Summary   400 System Error. Kindly try again or contact apacmulesupport@sg.panasonic.com
    for assistance.

    Error Details   System Error. Kindly try again or contact apacmulesupport@sg.panasonic.com
    for assistance.

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Subsystem was unavailable. Users were able to change passwords
    from subsequent requests. No action needed'
  - 'INC0059425 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Flow Direction

    Subsidiary

    Source System   SAP

    End System      SALES_EDI

    File Name       PLAP_INVOICE_ID20_20250530100023

    Storage Path

    API Name        sgst-audit-papi

    Error Source    PCS

    Transaction ID  68609e8a-f8be-4e4d-8bfe-a0c23a58da91

    Error Summary   500 Exception was found writing to file ''/outbound/pmi/sales_edi/fpl/PLAP_INVOICE_ID20_20250530100023_68609e8a-f8be-4e4d-8bfe-a0c23a58da91.txt''

    Error Details   Exception was found writing to file ''/outbound/pmi/sales_edi/fpl/PLAP_INVOICE_ID20_20250530100023_68609e8a-f8be-4e4d-8bfe-a0c23a58da91.txt''

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Error writing to PCS backup server. No action needed since
    file was posted to GITP'
- source_sentence: 'INC0060723 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Flow Direction

    Subsidiary

    Source System   SAP

    End System      SALES_EDI

    File Name       PLAP_INVOICE_ID20_20250612101014

    Storage Path

    API Name        sgst-audit-papi

    Error Source    PCS

    Transaction ID  bd7d54ee-4f11-45c7-94d5-b4c0561dbd3e

    Error Summary   500 Exception was found writing to file ''/outbound/pmi/sales_edi/fpl/PLAP_INVOICE_ID20_20250612101014_bd7d54ee-4f11-45c7-94d5-b4c0561dbd3e.txt''

    Error Details   Exception was found writing to file ''/outbound/pmi/sales_edi/fpl/PLAP_INVOICE_ID20_20250612101014_bd7d54ee-4f11-45c7-94d5-b4c0561dbd3e.txt''

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Error writing to PCS backup server. No action needed since
    file was posted to GITP'
  sentences:
  - INC0018095 Please generate missing FPL no. IJP2404099N_C1 which cannot be found
    in the shared drive and share to us urgently. Missing FPL invoices provided to
    user
  - 'INC0051933 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Interface       GITP

    Subsidiary      PIDSTH

    API Name        inds-global-if-mgmt-papi

    Flow Direction  Outbound

    Source System   SAP

    End System      PAGITP

    File Name       0000000002348362_9beb8c50-f7a3-11ef-a2e0-daed00c1551a.xml

    Storage Path    /INDS/prod/outbound/nocompany/purchase-order/0000000002348362_9beb8c50-f7a3-11ef-a2e0-daed00c1551a.xml

    Error Source    PA-GITP

    Transaction ID  9beb8c50-f7a3-11ef-a2e0-daed00c1551a

    Error Summary   500 RUNTIME_ERROR

    Error Details   ***********443/api/v1/purchase-order'' failed: internal server
    error (500).

    Comments        Unable to find 0000000002348362_9beb8c50-f7a3-11ef-a2e0-daed00c1551a.xml
    from Backup location


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Transaction has been reprocessed.'
  - 'INC0059426 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Flow Direction

    Subsidiary

    Source System   SAP

    End System      EPRO

    File Name       Y0GMM_ZAO0110_R41_ID20_20250530100056

    Storage Path

    API Name        sgst-audit-papi

    Error Source    PCS

    Transaction ID  53469570-bfd4-4d4e-8835-5394f9504fce

    Error Summary   500 Exception was found writing to file ''/outbound/pmi/epro/purchase-orders/Y0GMM_ZAO0110_R41_ID20_20250530100056_53469570-bfd4-4d4e-8835-5394f9504fce.txt''

    Error Details   Exception was found writing to file ''/outbound/pmi/epro/purchase-orders/Y0GMM_ZAO0110_R41_ID20_20250530100056_53469570-bfd4-4d4e-8835-5394f9504fce.txt''

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Error writing to PCS backup server. No action needed since
    file was posted to GITP'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
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
    "INC0060723 Dear Team\n\nError occurred while processing the EDI transaction. Please find the details below and attached is the file associated to the transaction.\n\nFlow Direction\nSubsidiary\nSource System   SAP\nEnd System      SALES_EDI\nFile Name       PLAP_INVOICE_ID20_20250612101014\nStorage Path\nAPI Name        sgst-audit-papi\nError Source    PCS\nTransaction ID  bd7d54ee-4f11-45c7-94d5-b4c0561dbd3e\nError Summary   500 Exception was found writing to file '/outbound/pmi/sales_edi/fpl/PLAP_INVOICE_ID20_20250612101014_bd7d54ee-4f11-45c7-94d5-b4c0561dbd3e.txt'\nError Details   Exception was found writing to file '/outbound/pmi/sales_edi/fpl/PLAP_INVOICE_ID20_20250612101014_bd7d54ee-4f11-45c7-94d5-b4c0561dbd3e.txt'\nComments\n\nNote: This is an automated mail, please do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic Error writing to PCS backup server. No action needed since file was posted to GITP",
    "INC0059426 Dear Team\n\nError occurred while processing the EDI transaction. Please find the details below and attached is the file associated to the transaction.\n\nFlow Direction\nSubsidiary\nSource System   SAP\nEnd System      EPRO\nFile Name       Y0GMM_ZAO0110_R41_ID20_20250530100056\nStorage Path\nAPI Name        sgst-audit-papi\nError Source    PCS\nTransaction ID  53469570-bfd4-4d4e-8835-5394f9504fce\nError Summary   500 Exception was found writing to file '/outbound/pmi/epro/purchase-orders/Y0GMM_ZAO0110_R41_ID20_20250530100056_53469570-bfd4-4d4e-8835-5394f9504fce.txt'\nError Details   Exception was found writing to file '/outbound/pmi/epro/purchase-orders/Y0GMM_ZAO0110_R41_ID20_20250530100056_53469570-bfd4-4d4e-8835-5394f9504fce.txt'\nComments\n\nNote: This is an automated mail, please do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic Error writing to PCS backup server. No action needed since file was posted to GITP",
    "INC0051933 Dear Team\n\nError occurred while processing the EDI transaction. Please find the details below and attached is the file associated to the transaction.\n\nInterface       GITP\nSubsidiary      PIDSTH\nAPI Name        inds-global-if-mgmt-papi\nFlow Direction  Outbound\nSource System   SAP\nEnd System      PAGITP\nFile Name       0000000002348362_9beb8c50-f7a3-11ef-a2e0-daed00c1551a.xml\nStorage Path    /INDS/prod/outbound/nocompany/purchase-order/0000000002348362_9beb8c50-f7a3-11ef-a2e0-daed00c1551a.xml\nError Source    PA-GITP\nTransaction ID  9beb8c50-f7a3-11ef-a2e0-daed00c1551a\nError Summary   500 RUNTIME_ERROR\nError Details   ***********443/api/v1/purchase-order' failed: internal server error (500).\nComments        Unable to find 0000000002348362_9beb8c50-f7a3-11ef-a2e0-daed00c1551a.xml from Backup location\n\nNote: This is an automated mail, please do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic Transaction has been reprocessed.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9979, 0.9960],
#         [0.9979, 1.0000, 0.9966],
#         [0.9960, 0.9966, 1.0000]])
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

* Size: 1,984 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                           | sentence_1                                                                           | label                                                          |
  |:--------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                               | string                                                                               | float                                                          |
  | details | <ul><li>min: 17 tokens</li><li>mean: 151.16 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 22 tokens</li><li>mean: 148.52 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.97</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | label            |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>INC0032084 PSV CS changed printer from HP LaserJet Pro MFP M225 (10.92.194.149) to ApeosPort-V C4476 (10.92.194.146) at Binh Duong warehouse (541). Kindly help to update. Thank you! Did the changes in PNQ system and the printer is working fine and the user confirmed the same. Will move to production using the SR REQ0031384</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                    | <code>INC0046692 Hi Team,  <br>  <br>Users are not able to print automatically from SAP. From the printer end it looks fine as it is prinitng normally and through PDF. Asc checked SapSprint service is also running. Can you please check the issue from your end. User confirmed the print is working fine in the system and asked to close the incident from our end.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | <code>1.0</code> |
  | <code>INC0051590 Nishtha faced error while opening and posting invoices in eInvoice. Please see attached. Dear @Bee Yen Loh san,  <br>Good day!!!  <br>  <br>As discussed and verified in teams call, there is no issue in the reported user (@Nishtha Singhal) login and able to approve the invoice documents in eWorkplace.  <br>Confirmed with your end, we are proceeding to close this ticket.  <br>  <br>Regards,  <br>Manoj.BK  <br>IT Consultant  <br>Panasonic Information Systems Company Asia Pacific (PISCAP)   <br>E-mail: manoj.bk@sg.panasonic.com</code>                                                                                                                                                                                                                                      | <code>INC0048617 unable to approve eSO refer attached screenshot Hi Abu san,  <br>Good evening!  <br>  <br>As discussed in teams, the eSO document approval was working as expected from your laptop and local IT team is checking on user's laptop issue,  <br>So, since this is user's laptop issue and local IT team is already verifying, we are proceeding to resolve/close this incident ticket 'INC0048617' upon confirmation.  <br>  <br>Thanks,  <br>Kannan B</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | <code>1.0</code> |
  | <code>INC0055598 Dear Team<br><br>Error occurred while processing the EDI transaction<br><br>Interface<br>Subsidiary      No Subsidiary<br>API Name        pana-pagitp-mgmt-eapi<br>Flow Direction  Inbound<br>Source System   PAGITP<br>End System      SAP S4Hana<br>File Name       No File Name<br>Storage Path    No Attachment<br>Error Source    Mulesoft<br>Transaction ID  9e252ef1-79ee-4371-bb07-5c99e90be686<br>Error Summary   500 SOURCE_RESPONSE_SEND<br>Error Details   Client connection was closed<br>Comments<br><br>Note: This is an automated mail, please do not reply.<br><br>Thanks and Regards,<br><br>APAC Support, Panasonic 1. Checked the log in Cloudwatch.  <br>2. Checked the file in common storage.  <br>3. Have verified that the data was processed in MBP.</code> | <code>INC0041910 Dear Team<br><br>Error occurred while processing the EDI transaction. Please find the details below and attached is the file associated to the transaction.<br><br>Interface       hriq-response<br>Subsidiary      PIDSAP<br>API Name        pana-hriq-mgmt-papi<br>Flow Direction  inbound<br>Source System   HRIQ<br>End System      eWork SNow System<br>File Name       No File Name<br>Storage Path    /INDS/prod/inbound/pidsap/daily-allowance/m3doxtx6n0autvre399_c1b07a55-2e0d-4fbe-806c-09b403701af9.dat<br>Error Source    SAP S/4Hana<br>Transaction ID  c1b07a55-2e0d-4fbe-806c-09b403701af9<br>Error Summary   400 BAD_REQUEST<br>Error Details   ***********44300/sap/opu/odata/sap/YMFI_SAPEWORK_API_SRV/inECLAIMSet/' failed: bad request (400).eWork Reference Number: eCLM-K1-2401775<br>Comments<br><br>Note: This is an automated mail, please do not reply.<br><br>Thanks and Regards,<br><br>APAC Support, Panasonic PIDSMY data which failed in SAP and is not posted to HRIQ</code> | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 1
- `fp16`: True
- `disable_tqdm`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
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
- `disable_tqdm`: True
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