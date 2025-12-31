---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:2678
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ["Dear Team\n\nError occurred while processing the EDI transaction\n\nInterface       SO-REPORT\nSubsidiary      PIDSMY\nAPI Name        ext-partners-order-mgmt-papi\nFlow Direction  Inbound\nSource System   SAP\nEnd System      NA\nFile Name       NA\nStorage Path    No Attachment\nError Source    Mulesoft\nTransaction ID  9cce66a6-83a5-43b0-b707-c96f11ae8052\nError Summary   501 RETRY_EXHAUSTED\nError Details   Path '/INDS/inbound/pidsmy/cpo/avnet/process/Avnet AS2 Summary Report_09300007.xls' doesn't exist\nComments\n\nNote: This is an automated mail, please do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic", 'User: Khine\r\nPls help to check why the DEFAULTCODE in partner function did not change to customer code'],
    ['We would like to inquire about FBL1N t-code as we observed that some Vendor Line Items does not show EWT while others have. Is there any FI Reports we could use to summarize the Vendor Line Items including the WT/EWT similar to FBL1N?\r\n\r\nNote: Please refer to the attached email for the details of the query.', 'ERROR PAGCS_GITP_GCS_PARTNER ([ Mon, 03 Mar 2025  14:48:48 +0800 ])  +++PROD+++'],
    ['server group does not exist or has too few resources', "Dear Team\n\nError occurred while processing the EDI transaction. Please find the details below and attached is the file associated to the transaction.\n\nInterface       goods-received-note\nSubsidiary      PIDSKR\nAPI Name        ext-partners-3pl-mgmt-papi\nFlow Direction  inbound\nSource System   Goods-received-notes\nEnd System      SAP S4Hana\nFile Name       KMTC5GCX20250310133333\nStorage Path    /INDS/inbound/pidskr/grn-confirmation/KMTC5GCX20250310133333_96727963-fd6a-11ef-a67a-964de83787c1.txt\nError Source    Mulesoft\nTransaction ID  96727963-fd6a-11ef-a67a-964de83787c1\nError Summary   501 RETRY_EXHAUSTED\nError Details   Path '/pidskr/kr_kmtc/ctl_l/KMTC5GCX20250310133333.RC' doesn't exist\nComments\n\nNote: This is an automated mail, please do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic"],
    ["Error occurred while processing the EDI transaction\r\nInterface\tIPS31\r\nSubsidiary\tGITP\r\nAPI Name\tinds-global-if-mgmt-papi\r\nFlow Direction\toutbound\r\nSource System\tGITP\r\nEnd System\tSAP\r\nFile Name\tPIDSMY60230921300000000001_ips31_314_01_001_240720241543\r\nStorage Path\tNo Attachment\r\nError Source\tSAP\r\nTransaction ID\t7e1478e0-49d4-11ef-a213-0eec66d962d1\r\nError Summary\t500 INTERNAL_SERVER\r\nError Details\t***********443/api/v1/purchase-orders' failed: Timeout exceeded.", 'Normally after the selection of CN Type & Order Reason the bottom screen should appear some option/selection like Amount to refund or Qty to refund. Mr. Dinesh were supposed to follow up on this case which was reported to him on 19/8/2025. So far no advice yet.'],
    ["Dear Team\n\nError occurred while processing the EDI transaction. Please find the details below and attached is the file associated to the transaction.\n\nFlow Direction\nSubsidiary\nSource System   SAP\nEnd System      EPRO\nFile Name       Y0GMM_ZAO0110_R41_ID20_20250609090008\nStorage Path\nAPI Name        sgst-audit-papi\nError Source    PCS\nTransaction ID  faade8ae-7741-42d1-b2c1-d97553034a7c\nError Summary   500 Exception was found writing to file '/outbound/pmi/epro/purchase-orders/Y0GMM_ZAO0110_R41_ID20_20250609090008_faade8ae-7741-42d1-b2c1-d97553034a7c.txt'\nError Details   Exception was found writing to file '/outbound/pmi/epro/purchase-orders/Y0GMM_ZAO0110_R41_ID20_20250609090008_faade8ae-7741-42d1-b2c1-d97553034a7c.txt'\nComments\n\nNote: This is an automated mail, please do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic", "User : Minami \r\n\r\nSR ticket RITM0059998 was completed but till date,  user didn't receive any notification of their registration and unable to log into their SF."],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    "Dear Team\n\nError occurred while processing the EDI transaction\n\nInterface       SO-REPORT\nSubsidiary      PIDSMY\nAPI Name        ext-partners-order-mgmt-papi\nFlow Direction  Inbound\nSource System   SAP\nEnd System      NA\nFile Name       NA\nStorage Path    No Attachment\nError Source    Mulesoft\nTransaction ID  9cce66a6-83a5-43b0-b707-c96f11ae8052\nError Summary   501 RETRY_EXHAUSTED\nError Details   Path '/INDS/inbound/pidsmy/cpo/avnet/process/Avnet AS2 Summary Report_09300007.xls' doesn't exist\nComments\n\nNote: This is an automated mail, please do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic",
    [
        'User: Khine\r\nPls help to check why the DEFAULTCODE in partner function did not change to customer code',
        'ERROR PAGCS_GITP_GCS_PARTNER ([ Mon, 03 Mar 2025  14:48:48 +0800 ])  +++PROD+++',
        "Dear Team\n\nError occurred while processing the EDI transaction. Please find the details below and attached is the file associated to the transaction.\n\nInterface       goods-received-note\nSubsidiary      PIDSKR\nAPI Name        ext-partners-3pl-mgmt-papi\nFlow Direction  inbound\nSource System   Goods-received-notes\nEnd System      SAP S4Hana\nFile Name       KMTC5GCX20250310133333\nStorage Path    /INDS/inbound/pidskr/grn-confirmation/KMTC5GCX20250310133333_96727963-fd6a-11ef-a67a-964de83787c1.txt\nError Source    Mulesoft\nTransaction ID  96727963-fd6a-11ef-a67a-964de83787c1\nError Summary   501 RETRY_EXHAUSTED\nError Details   Path '/pidskr/kr_kmtc/ctl_l/KMTC5GCX20250310133333.RC' doesn't exist\nComments\n\nNote: This is an automated mail, please do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic",
        'Normally after the selection of CN Type & Order Reason the bottom screen should appear some option/selection like Amount to refund or Qty to refund. Mr. Dinesh were supposed to follow up on this case which was reported to him on 19/8/2025. So far no advice yet.',
        "User : Minami \r\n\r\nSR ticket RITM0059998 was completed but till date,  user didn't receive any notification of their registration and unable to log into their SF.",
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
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

* Size: 2,678 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                         | sentence_1                                                                                        | label                                                          |
  |:--------|:---------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                             | string                                                                                            | float                                                          |
  | details | <ul><li>min: 11 characters</li><li>mean: 479.65 characters</li><li>max: 11518 characters</li></ul> | <ul><li>min: 8 characters</li><li>mean: 417.36 characters</li><li>max: 11518 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.13</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | label            |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Dear Team<br><br>Error occurred while processing the EDI transaction<br><br>Interface       SO-REPORT<br>Subsidiary      PIDSMY<br>API Name        ext-partners-order-mgmt-papi<br>Flow Direction  Inbound<br>Source System   SAP<br>End System      NA<br>File Name       NA<br>Storage Path    No Attachment<br>Error Source    Mulesoft<br>Transaction ID  9cce66a6-83a5-43b0-b707-c96f11ae8052<br>Error Summary   501 RETRY_EXHAUSTED<br>Error Details   Path '/INDS/inbound/pidsmy/cpo/avnet/process/Avnet AS2 Summary Report_09300007.xls' doesn't exist<br>Comments<br><br>Note: This is an automated mail, please do not reply.<br><br>Thanks and Regards,<br><br>APAC Support, Panasonic</code> | <code>User: Khine  <br>Pls help to check why the DEFAULTCODE in partner function did not change to customer code</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | <code>0.0</code> |
  | <code>We would like to inquire about FBL1N t-code as we observed that some Vendor Line Items does not show EWT while others have. Is there any FI Reports we could use to summarize the Vendor Line Items including the WT/EWT similar to FBL1N?  <br>  <br>Note: Please refer to the attached email for the details of the query.</code>                                                                                                                                                                                                                                                                                                                                                                        | <code>ERROR PAGCS_GITP_GCS_PARTNER ([ Mon, 03 Mar 2025  14:48:48 +0800 ])  +++PROD+++</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | <code>0.0</code> |
  | <code>server group does not exist or has too few resources</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | <code>Dear Team<br><br>Error occurred while processing the EDI transaction. Please find the details below and attached is the file associated to the transaction.<br><br>Interface       goods-received-note<br>Subsidiary      PIDSKR<br>API Name        ext-partners-3pl-mgmt-papi<br>Flow Direction  inbound<br>Source System   Goods-received-notes<br>End System      SAP S4Hana<br>File Name       KMTC5GCX20250310133333<br>Storage Path    /INDS/inbound/pidskr/grn-confirmation/KMTC5GCX20250310133333_96727963-fd6a-11ef-a67a-964de83787c1.txt<br>Error Source    Mulesoft<br>Transaction ID  96727963-fd6a-11ef-a67a-964de83787c1<br>Error Summary   501 RETRY_EXHAUSTED<br>Error Details   Path '/pidskr/kr_kmtc/ctl_l/KMTC5GCX20250310133333.RC' doesn't exist<br>Comments<br><br>Note: This is an automated mail, please do not reply.<br><br>Thanks and Regards,<br><br>APAC Support, Panasonic</code> | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16

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
- `num_train_epochs`: 3
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
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 2.9762 | 500  | 0.1252        |


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