---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:14896
- loss:MarginMSELoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: who is cs phu hung
  sentences:
  - 'Requester : Ms.Lee Junghwa (lee.junghwa@kr.panasonic.com) Please check GI Pending
    ERROR. SO#: 310005117 Delivery#: 355003218 Pland Gds Mvmnt Date : 2024-06-04 GI
    Pending : Load Summary Error( E) Still pending GI. Could you advice for me?'
  - 'Salesman name: - Vo Quy Hoang (south) - Nguy?n Qu?c Th?nh (north) for South salesman,
    same approver level as Pham Van Cuong for North salesman, same approver level
    as Pham Van Hanh'
  - Pls check we cannot find CS PHUC HUNG in Partner Branch name
- source_sentence: do you have to approve dmr
  sentences:
  - We were trying to add Z2 (discontinue status) via DMR for material N2QAYB001008,
    but even after approval, no changes were made. DChain-spec. status is still empty
    in MM03. Please help immediately, because we keep ordering discontinued materials
    without knowing (through ZMRP). For Detail Please see ServiceNow attachmant and
    My Email Too.
  - 'We checked and found 4 SCMM PO s sent to SAP last 26 Mar still does not have
    SAP PO numbers. Please help to check. PPH: O0006002 O0006001 PMSS: B0096892 B0096890'
  - For your kind assistance re DMR, no pending approval but still partially posted.
    See attached screenshot.
- source_sentence: what is the nsc part number
  sentences:
  - 'PI-252466 With PGI-SO-2405001853 if we check Why Product Name is : CS_SERVICE
    ? Because we don''t know this and we check in SO part is : A50C2401 , (Product
    Name Should be A50C2401), please fix this'
  - 'The Service Center (NSC) has not yet carried out the Unpaid Process, but PASS
    (ASC) has received (User Mistake), so the generated invoiced button can no longer
    be clicked and has entered the SC (NSC) inventory, For this reason, what is the
    status of this part: 1).Part Code AXW-12138000006399, Order Number PGI-SO-2405001811,
    Qty =1 So, Kindly please help change this SO (Part) to be Invoiced, Thanks'
  - PM-WO-240700895 _WOLI WITH PRICING ALTHOUGH UNDER WARRANTY - Please assist to
    rectify issue to FOC. / under warranty.
- source_sentence: error occurred while processing the edi transaction
  sentences:
  - Dear Team Error occurred while processing the EDI transaction. Please find the
    details below and attached is the file associated to the transaction. Interface
    employee-minimaster-ridm Subsidiary PAPAMY API Name pana-global-hriq-sapi Flow
    Direction inbound Source System successFactors End System RIDM File Name ridm-empminimaster_10dc3880-9bd2-11ef-b087-da452223c681.json
    Storage Path /PAPAMY/ridm/employee-mini-master/ridm-empminimaster_10dc3880-9bd2-11ef-b087-da452223c681.json
    Error Source RIDM Transaction ID
  - we have a new NonTrade business under C1S4, but we don't know how to start the
    new business, please guide us how to do in SAP.
  - Dear Team Error occurred while processing the EDI transaction. Please find the
    details below and attached is the file associated to the transaction. Interface
    hriq-response Subsidiary PIDSAP API Name pana-hriq-mgmt-papi Flow Direction inbound
    Source System HRIQ End System eWork SNow System File Name No File Name Storage
    Path /INDS/prod/inbound/pidsap/daily-allowance/m3hz87lwt7eczr3dfgl_a18d29de-931c-4e3f-8d45-b0672ee114e0.dat
    Error Source SAP S/4Hana Transaction ID a18d29de-931c-4e3f-8d45-b0672ee114e0 Error
- source_sentence: cannot upload due to new model registered
  sentences:
  - 'Item Part PGVF3191ZAC1 is cannot be uploaded to the DMR, When we try to Upload
    DMR Template is appear notification "Enter a valid material" Message No. /IRM/EPG107,
    For detail Please see my email and ITMAAS (ServiceNow) Attachment *Surya San (Support
    Team SAP) Resolved Explaination : The Model is not synced into DMR , since the
    error is causing to do any kind of postings (Already Done).'
  - 'User: Bhernadette Campos Issue: eCD-S1-24000508 was approved and posted . However,
    no generated PIDSAP Debit note file that can be downloaded.'
  - Need to re-upload due new model registered.
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
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'MPNetModel'})
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
    'cannot upload due to new model registered',
    'Need to re-upload due new model registered.',
    'Item Part PGVF3191ZAC1 is cannot be uploaded to the DMR, When we try to Upload DMR Template is appear notification "Enter a valid material" Message No. /IRM/EPG107, For detail Please see my email and ITMAAS (ServiceNow) Attachment *Surya San (Support Team SAP) Resolved Explaination : The Model is not synced into DMR , since the error is causing to do any kind of postings (Already Done).',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.9958, -0.9956],
#         [ 0.9958,  1.0000, -0.9962],
#         [-0.9956, -0.9962,  1.0000]])
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

* Size: 14,896 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, <code>sentence_2</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                       | sentence_1                                                                         | sentence_2                                                                        | label                                                              |
  |:--------|:---------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                             | string                                                                            | float                                                              |
  | details | <ul><li>min: 5 tokens</li><li>mean: 9.49 tokens</li><li>max: 31 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 71.63 tokens</li><li>max: 239 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 60.1 tokens</li><li>max: 249 tokens</li></ul> | <ul><li>min: -4.92</li><li>mean: 9.74</li><li>max: 21.81</li></ul> |
* Samples:
  | sentence_0                                                 | sentence_1                                                                                                                                                                                                | sentence_2                                                                                                                                                                                                                                | label                           |
  |:-----------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------|
  | <code>how long does it take for something to change</code> | <code>STATUS ALSO NEED TAKE SO LONG TIME TO CHANGE TO DELIVED, CUSTOMER REALLY CAN NOT WAIT 20 MIN - 30 MIN JUST FOR THE INVOICE ONLY, THEY ALSO GOT THEIR THING AND ALSO NEED ARRANGE THEIR THING</code> | <code>User: Waimun Pls help to check why system calculate the DA days incorrect. Pls note the eTravel already posted and DA days have been corrected via the Actual no of days.</code>                                                    | <code>13.813886404037476</code> |
  | <code>what is lan fatt pmso code</code>                    | <code>Dealer Lian Fatt PM-SO-2406000674. Request to resync order of parts to SAP.</code>                                                                                                                  | <code>Refer to RITM0021180 for more details with attachments. User is from PMDC. Unable to locate the reported missing DOs in FMS. Require SOA team help to check if documents have flowed in SOA platform.</code>                        | <code>9.000596761703491</code>  |
  | <code>what is the net value for ir</code>                  | <code>There is a clear difference between the IR amount and Net value but system showing as 0.00</code>                                                                                                   | <code>Asset no: 4100000002, Company code: 7100 created in PNP Scenario: Asset to be capitalized in Apr 2024, Depreciation for Apr 2024: to include Apr 23 - Mar 24 + Apr 24 Cost of asset: USD 83,962.32 Monthly dep: USD 1,749.22</code> | <code>17.671729564666748</code> |
* Loss: [<code>MarginMSELoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#marginmseloss)

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `max_steps`: 931
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
- `max_steps`: 931
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

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.5371 | 500  | 122.7352      |


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

#### MarginMSELoss
```bibtex
@misc{hofstätter2021improving,
    title={Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation},
    author={Sebastian Hofstätter and Sophia Althammer and Michael Schröder and Mete Sertkan and Allan Hanbury},
    year={2021},
    eprint={2010.02666},
    archivePrefix={arXiv},
    primaryClass={cs.IR}
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