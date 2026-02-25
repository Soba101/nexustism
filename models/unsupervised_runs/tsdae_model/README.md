---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:4000
- loss:DenoisingAutoEncoderLoss
base_model: google-bert/bert-base-uncased
widget:
- source_sentence: NLS_FMS Interface LT3-15057
  sentences:
  - NLS_FMS Interface Error LT3-15057
  - Dear Team Error occurred while processing the EDI transaction. Please find the
    details below and attached is the file associated to the transaction. Interface
    GITP Subsidiary PIDSAP API Name inds-global-if-mgmt-papi Flow Direction Outbound
    Source System SAP End System IBMMQ File Name 0000000001727283_0684bc60-ad3c-11ef-bd0c-96a5d4410501.xml
    Storage Path /INDS/prod/outbound/nocompany/ordrsp/0000000001727283_0684bc60-ad3c-11ef-bd0c-96a5d4410501.xml
    Error Source Mulesoft Transaction ID 0684bc60-ad3c-11ef-bd0c-
  - 'Seeking your help to check Expense Charge Rate : Storage Fee under Export Billing,
    why the container details not displaying upon entry. Kindly check attached file
    for the testing that we had done in THQ 304.'
- source_sentence: refer attachment for both part under WOLi 00000003 & 00000004 already
    and we already repair the seek your to change status part from Dlv Delivered for
    us to close the.
  sentences:
  - PAPVN Local IT haven't received any E-Invoice Outbound Error Notifications.
  - PNQ910 showing error when user is trying to access SE16N
  - refer to attachment below, for both part under WOLi 00000003 & 00000004 already
    delivered and we already repair the set.kindly seek your help to change the status
    part from Dlv Processing to Delivered for us to close the job. thanks.
- source_sentence: Pls cancel WO PSV-WO-2405013656 &
  sentences:
  - PSV-WO-2409009008, can not assign technician for this WO, pls help us check &
    fix. When we try to assign technician, the information in the technician in category
    of WO is blank. Pls check & fix. Thank you
  - Pls cancel WO PSV-WO-2405013656 & PSV-WO-2404013475
  - 'In ICGenerateWarrantyRegistrationEmail bill, there are 3 date fields: Ng y ??ng
    k ; Ng y mua h ng; Ng y h?t h?n which are in English language. Please change them
    into Vietnamese in "dd/mm/yyyy" order.'
- source_sentence: 'Dear Team Error occurred processing EDI transaction . find the
    details below and attached is associated to the transaction Flow Subsidiary Source
    SAP End einvoice File Name Storage Path /inbound/SAP/eInvoice/0000000000049086_To_einvoice.json
    API Name sgst-fi-invoice-papi Error Source INVOICE Transaction ID 87a10cd0-c665-11ef-a3ad-a699d2f48aa7
    Summary 500 HTTP POST on resource failed: internal server error (Er'
  sentences:
  - These are the paid order but doesn t push to OMS, and we noticed the last order
    pushed to SAP is Jun 6, 11:14 AM. Can you please check and resolve this issue
    by today? 9000007503 9000007506 9000007509 9000007512 9000007515 9000007518 9000007521
    9000007524 9000007527 9000007530 9000007536 9000007539 9000007542 9000007545 9000007551
  - Dear AMS team, PAPVN-TL2 WH member faced an issue with Lot no. display in WM label.
    Detail is in the attached file. Please help to check it. Brgs, Chung
  - 'Dear Team Error occurred while processing the EDI transaction. Please find the
    details below and attached is the file associated to the transaction. Flow Direction
    outbound Subsidiary PAPVN-TL2 Source System SAP End System einvoice File Name
    0000000000049086 Storage Path /inbound/SAP/eInvoice/0000000000049086_To_einvoice.json
    API Name sgst-fi-invoice-papi Error Source INVOICE Transaction ID 87a10cd0-c665-11ef-a3ad-a699d2f48aa7
    Error Summary 500 HTTP POST on resource '' failed: internal server error (500).
    Er'
- source_sentence: 'User reported below: 1) Some CPO found in Summary report but not
    in SO report (6208892383) 2) found WM SO report but Summary report (6208892493
    6208892434'
  sentences:
  - User reported issues below:- 1) Some CPO found in Summary report , but not in
    WM SO report (6208892383) 2) Some found in WM SO report, but not found in Summary
    report (6208892493 & 6208892434 & 6208892379)
  - Kindly assist this case job in warranty but appear amount 5000013210 WALKS IN
    CUSTOMER (JB) PCMJI240801066 30.08.2024 15.09.2024 Y2 9237101224C042 MYR 55.41
  - 'User : Caremen DO 551008915 block for delivery but user confirmed the credit
    limit granted is able to cover. BP : 80935820 Pls kindly help to check the credit
    exposure .'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on google-bert/bert-base-uncased

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) <!-- at revision 86b5e0934494bd15c9632b12f734a8a67f723594 -->
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
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
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
    'User reported below: 1) Some CPO found in Summary report but not in SO report (6208892383) 2) found WM SO report but Summary report (6208892493 6208892434',
    'User reported issues below:- 1) Some CPO found in Summary report , but not in WM SO report (6208892383) 2) Some found in WM SO report, but not found in Summary report (6208892493 & 6208892434 & 6208892379)',
    'User : Caremen DO 551008915 block for delivery but user confirmed the credit limit granted is able to cover. BP : 80935820 Pls kindly help to check the credit exposure .',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9983, 0.9662],
#         [0.9983, 1.0000, 0.9618],
#         [0.9662, 0.9618, 1.0000]])
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

* Size: 4,000 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             |
  | details | <ul><li>min: 3 tokens</li><li>mean: 52.88 tokens</li><li>max: 201 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 74.78 tokens</li><li>max: 246 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                             | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>ticket [Ref #] opened on behalf Angie Chan . Pls kindly help check why for "Post Amount" Daily Allowance" is? And", the "Accommodation Employee" is zero?</code>                                                                                                                                                                                 | <code>Transferred ticket [Ref # 300-258675], opened on behalf of Angie Chan. Pls kindly help to check why for "Post Amount", the "Daily Allowance" is zero? And for "Pre Amount", the "Accommodation for Employee" is zero?</code>                                                                                                                                                                                                                                                                                                            |
  | <code>material GR in List is because to wrong price, after cancel we to receive SAP after fixed we have for the (Receiving Error) (Detail Please attachment)</code>                                                                                                                                                                                    | <code>PO and material GR in the List is already cancel because to wrong price, after cancel we want to receive back from SAP after price are fixed, but we have problem for the receiving (Receiving Error) (For Detail Please see the attachment)</code>                                                                                                                                                                                                                                                                                     |
  | <code>Team occurred processing the EDI Interface Subsidiary PIDSAP Name ext-partners-order-mgmt-papi Flow Direction System SAP End System NA File Name NA Storage Path Error Source ID Summary 500 UNKNOWN Details parameter 'path' was assigned with' vars.varINITVariables . "so-report_initial-backup-path"]' which to Required need to be a</code> | <code>Dear Team Error occurred while processing the EDI transaction Interface SO-REPORT Subsidiary PIDSAP API Name ext-partners-order-mgmt-papi Flow Direction Inbound Source System SAP End System NA File Name NA Storage Path No Attachment Error Source Mulesoft Transaction ID dc1ed670-79ca-4a7c-b0ab-9fead9aaca90 Error Summary 500 UNKNOWN Error Details Required parameter 'path' was assigned with value '#[vars.varINITVariables."so-report_initial-backup-path"]' which resolved to null. Required parameters need to be a</code> |
* Loss: [<code>DenoisingAutoEncoderLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#denoisingautoencoderloss)

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `do_predict`: False
- `eval_strategy`: no
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

#### DenoisingAutoEncoderLoss
```bibtex
@inproceedings{wang-2021-TSDAE,
    title = "TSDAE: Using Transformer-based Sequential Denoising Auto-Encoderfor Unsupervised Sentence Embedding Learning",
    author = "Wang, Kexin and Reimers, Nils and Gurevych, Iryna",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    pages = "671--688",
    url = "https://arxiv.org/abs/2104.06979",
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