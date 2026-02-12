---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:976
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: routing no accounts even thought in routing master there is ak2
    and ao gm applicant. advise,
  sentences:
  - pacmy-sales order-2403003455 pacmy-sales order-2403003524
  - psv-wo-2404009633--> psv-sales order-2404009232 does not link to sap,
  - routing no accounts even thought in routing master there is ak2 and ao gm applicant.
    advise,
- source_sentence: actual • costs transferred to a different work order/sales order
    are not always going to sap • cloned lines to remove cost from approved/submitted
    timesheet entries that can't be reversed not always going to sap • goods return
    doesn’t work so need to reverse cost on sales order in salesforce • in salesforce
    the costs and margins look correct and the transactions appear to have been successful
    expected • negative amounts to be sent to sap when there is negative cost • manual
    entries will sync to sap when sales order is invoiced or closed • costs and margins
    to be the same in both salesforce and sap impact • total cost not sent to sap
    $146,112.95 as at 11/07/2023 • mismatch between salesforce and sap examples •
    work order 00007263 costs were transferred to work order 00020739 as the work
    order needed to be under a different profit and cost centre. in salesforce work
    order 00007263 shows $0 cost and sales but in sap work order 00020739 only shows
    the purchase order that were moved • work order 00026119 sa-49965 was amended
    after the timesheet entry was submitted as the hours were on the incorrect work
    order, new sa-70479
  sentences:
  - actual • costs transferred to a different work order/sales order are not always
    going to sap • cloned lines to remove cost from approved/submitted timesheet entries
    that can't be reversed not always going to sap • goods return doesn’t work so
    need to reverse cost on sales order in salesforce • in salesforce the costs and
    margins look correct and the transactions appear to have been successful expected
    • negative amounts to be sent to sap when there is negative cost • manual entries
    will sync to sap when sales order is invoiced or closed • costs and margins to
    be the same in both salesforce and sap impact • total cost not sent to sap $146,112.95
    as at 11/07/2023 • mismatch between salesforce and sap examples • work order 00007263
    costs were transferred to work order 00020739 as the work order needed to be under
    a different profit and cost centre. in salesforce work order 00007263 shows $0
    cost and sales but in sap work order 00020739 only shows the purchase order that
    were moved • work order 00026119 sa-49965 was amended after the timesheet entry
    was submitted as the hours were on the incorrect work order, new sa-70479
  - check 2 email regostered but cannot login. sasiwimon.thaweephon@th.panasonic.computer
    rudklao.trongwiwat@th.panasonic.computer
  - return note (dpr) 1.12394 2.12396 part hanging at counter.
- source_sentence: help to check reason and fix error of not sync customer code 5000027030
    of psv from dmr to sap
  sentences:
  - unable to change the valuation class for the part number :w035a-9bf00 from 7900
    in house to 3000 raw material
  - help to change nominal invoiced from rp.172,000.00 (seratus tujuh puluh dua ribu
    rupiah) to be 172,050.00 (seratus tujuh puluh dua ribu lima puluh rupiah) in pgi-wo-2404000597
    / pgiji-2404-02866 the nominal should be same with proforma invoice new pgipj-2404-00065
    for detail see servicenow attachment and my email
  - help to check reason and fix error of not sync customer code 5000027030 of psv
    from dmr to sap
- source_sentence: check we cannot see the extended warranty sales (ews) order psv-ews-2405000358
    in sap and v-invoice
  sentences:
  - check we cannot see the extended warranty sales (ews) order psv-ews-2405000358
    in sap and v-invoice
  - 'check the below case as one part has been excess consumed comparison to production
    order qty. we need to know how this happened in sap? part no: arbddk100390 order
    no: 106594551 order qty: 606 nos actual consumed: 894 nos'
  - 'user: sofia loraine issue: eso-s1-24005475 posting failed message": "sales order
    110038198 is not yet send to ework",'
- source_sentence: 'http post on resource ''https://sgst-fi-invoice-einvoice-sapi-qoq0kf.internal-hnygb7.sgp-s1.cloudhub.io:443/api/v1/create-invoice''
    failed: timeout exceeded.'
  sentences:
  - 'when linking account onto genesis, magento didn''t consider existing account
    but instead, created another customer account. this leads to the issue of duplicate
    customer account with 1 phone number. therefore, when we register warranty for
    customers, the error "can not transfer data to genesis" occurs. here is an example
    you can check: 84974856380 tell us the root cause and solutions for this issue.
    this is so urgent and occurs frequently with around 30 cases recorded. !'
  - 'http post on resource ''https://sgst-fi-invoice-einvoice-sapi-qoq0kf.internal-hnygb7.sgp-s1.cloudhub.io:443/api/v1/create-invoice''
    failed: timeout exceeded.'
  - 551009487 edo error message update control of movement type is incorrect (entry
    687 x x e l _ e) posting to sap error help to check urgently as it is related
    to month end closing
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision e8c3b32edf5434bc2275fc9bab85f82640a19130 -->
- **Maximum Sequence Length:** 128 tokens
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
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False, 'architecture': 'MPNetModel'})
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
    "http post on resource 'https://sgst-fi-invoice-einvoice-sapi-qoq0kf.internal-hnygb7.sgp-s1.cloudhub.io:443/api/v1/create-invoice' failed: timeout exceeded.",
    "http post on resource 'https://sgst-fi-invoice-einvoice-sapi-qoq0kf.internal-hnygb7.sgp-s1.cloudhub.io:443/api/v1/create-invoice' failed: timeout exceeded.",
    '551009487 edo error message update control of movement type is incorrect (entry 687 x x e l _ e) posting to sap error help to check urgently as it is related to month end closing',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 1.0000, 0.0241],
#         [1.0000, 1.0000, 0.0241],
#         [0.0241, 0.0241, 1.0000]])
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

* Size: 976 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 976 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             |
  | details | <ul><li>min: 6 tokens</li><li>mean: 49.87 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 49.87 tokens</li><li>max: 128 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>sales order: psv-sales order-2405005260, psv-sales order-2405005261, psv-sales order-2405005272, psv-sales order-2405005316, psv-sales order-2405005318, psv-sales order-2405005327 these above sos have hours process in system less than 24 hours. but psr 24h report shows status "fail" for these sos. check and correct the status in report as soon as possible</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | <code>sales order: psv-sales order-2405005260, psv-sales order-2405005261, psv-sales order-2405005272, psv-sales order-2405005316, psv-sales order-2405005318, psv-sales order-2405005327 these above sos have hours process in system less than 24 hours. but psr 24h report shows status "fail" for these sos. check and correct the status in report as soon as possible</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
  | <code>can't select non bp type/category in eiaf19876</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>can't select non bp type/category in eiaf19876</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
  | <code>all, assist to check this idoc error is referring to currency amount. provide more information as the transaction is in usd currency. 'required field 6x4a00573571 was not transferred in parameter currencyamount' -----original message----- from: job step user for pcoa <piscap-basis-team@sg.panasonic.computer> sent: tuesday, april 30, 2024 4:03 am to: piscap_nsc_ams_fi <nsc-ams-fi@sg.panasonic.computer>; piscap_smap_support <smap.support@sg.panasonic.computer>; cynthia chii wang kho <cynthia.khocw@sg.panasonic.computer>; siti hawa <siti.hawa@sg.panasonic.computer> subject: [piscap_smap_support:233686] smapmntrg interface daily monitoring report hi, this is idoc system generated notification. for 29.04.2024 transaction from smapmntrg to sap 7100 find the detail error idoc no. and smapmntrg refernce no. in attached file. this is idoc system generated notification.</code> | <code>all, assist to check this idoc error is referring to currency amount. provide more information as the transaction is in usd currency. 'required field 6x4a00573571 was not transferred in parameter currencyamount' -----original message----- from: job step user for pcoa <piscap-basis-team@sg.panasonic.computer> sent: tuesday, april 30, 2024 4:03 am to: piscap_nsc_ams_fi <nsc-ams-fi@sg.panasonic.computer>; piscap_smap_support <smap.support@sg.panasonic.computer>; cynthia chii wang kho <cynthia.khocw@sg.panasonic.computer>; siti hawa <siti.hawa@sg.panasonic.computer> subject: [piscap_smap_support:233686] smapmntrg interface daily monitoring report hi, this is idoc system generated notification. for 29.04.2024 transaction from smapmntrg to sap 7100 find the detail error idoc no. and smapmntrg refernce no. in attached file. this is idoc system generated notification.</code> |
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

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
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
- `num_train_epochs`: 1
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