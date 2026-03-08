---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:2737
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L12-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L12-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2) <!-- at revision 7b0235231ca2674cb8ca8f022859a6eba2b1c968 -->
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
    ['cause : pgi - so - 2505003793 is appear in sfdc & partner portal but not appear ( cannot be read in sap ), please fix. | cat : application / software', "effect : job : pm - wo - 2501004489 part no : m4d154810zm ( pcb ) part still udr part requested - didn't link to sap pls assist to link as customer request urgent repair / to close job sheet | cat : application / software | delta _ hours : 3266. 45"],
    ["cause : pls cancel ews psv - ews - 2503000221 in sfdc and don't trigger to sap | cat : application / software", 'effect : dear team error occurred while processing the edi transaction interface subsidiary no subsidiary api name pana - pagitp - mgmt - eapi flow direction inbound source system pagitp end system sap s4hana file name no file name storage path no attachment error source mulesoft transaction id 946bd1f9 - c3a8 - 4309 - 9216 - df00e3efc6c4 error summary 500 source _ response _ send error details client connection was closed comments note : this is an automated mail, please do not reply. thanks and regards, apac support, panasonic | cat : application / software | delta _ hours : 2826. 65'],
    ['cause : material : srf3704 - kd10 | cat : application / software', 'effect : dear team error occurred while processing the edi transaction interface change - over subsidiary pau api name pana - sdesk - ext - eapi flow direction outbound source system pau end system zoho file name no file name storage path no attachment error source pana - sdesk - ext - eapi transaction id 819e3530 - a313 - 11ef - 8b0c - 02b4130d4440 error summary 500 expression error details expression : " org. mule. runtime. api. exception. muleruntimeexception - exception was found trying to retrieve the contents of file / zoho / live / outbound / status / remote / co _ status _ 20241115163317536699. csv org. mule. runtime. api. exception. muleruntimeexception : exception was found trying to retrieve the contents of file / zoho / live / outbound / status / remote / co _ status _ 20241115163317536699. csv caused by : sftp error ( ssh _ fx _ failure ) : failure. at org. apache. sshd. sftp. client. impl. abstractsftpclient. throwstatusexception ( abstractsftpclient. java : 277 ) at org. apache. sshd. sftp. client. impl. abstractsftpclient. checkhandleresponse ( abstractsftpclient. java : 299 ) at org. apache. sshd. sftp. client. impl. abstractsftpclient. checkhandle ( abstractsftpclient. java : 290 ) at org. apache. sshd. sftp. client. impl. abstractsftpclient. open ( abstractsftpclient. java : 589 ) at org. apache. sshd. sftp. client. impl. sftpinputstreamasync. ( sftpinputstreamasync. java : 75 ) at org. apache. sshd. sftp. client. impl. abstractsftpclient. read ( abstractsftpclient. java : 1196 ) at org. apache. sshd. sftp. client. sftpclient. read ( sftpclient. java : 909 )'],
    ['cause : dear team multiple errors occurred while processing the edi transaction. please see attached file for more details. error count 14 note : this is an automated mail, please do not reply. thanks and regards, apac support, panasonic | cat : application / software', 'effect : abap logic inquiry for the rr interface | cat : application / software | delta _ hours : 719. 37'],
    ['cause : please check wo psv - wo - 2410002855 wrong tax amount | cat : application / software', 'effect : dear team error occurred while processing the edi transaction interface change - over subsidiary pau api name pana - sdesk - ext - eapi flow direction outbound source system pau end system zoho file name no file name storage path no attachment error source pana - sdesk - ext - eapi transaction id 56fcce20 - 39ce - 11f0 - 8032 - eee27e3a0f38 error summary 500 expression error details " expression : \\ " org. mule. runtime. api. exception. muleruntimeexception - could not obtain connection to fetch file / zoho / live / outbound / status / remote / co _ status _ 20250526110806711817. csv \\ norg. mule. runtime. api. exception. muleruntimeexception : could not obtain connection to fetch file / zoho / live / outbound / status / remote / co _ status _ 20250526110806711817. csv \\ ncaused by : org. mule. runtime. api. connection. connectionexception : could not establish sftp connection with host :\'10. 86. 48. 62\'at port :\'22\'- session. connect : java. net. sockettimeoutexception : read timed out \\ ncaused by : com. jcraft. jsch. jschexception : session. connect : java. net. sockettimeoutexception : read timed out \\ n \\ tat com. jcraft. jsch. session. connect ( session. java : 565 ) \\ n \\ tat com. jcraft. jsch. session. connect ( session. java : 183 ) \\ n \\ tat org. mule. extension. sftp. internal. connection. sftpclient. connect ( sftpclient. java : 182 ) \\ n \\ tat org. mule. extension. sftp. internal. connection. sftpclient. login ( sftpclient. java : 164 ) \\ n \\ tat org. mule. extension. sftp. internal. connection. sftpconnectionprovider. connect ( sftpconnectionprovider. java : 141 ) \\ n \\ tat'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'cause : pgi - so - 2505003793 is appear in sfdc & partner portal but not appear ( cannot be read in sap ), please fix. | cat : application / software',
    [
        "effect : job : pm - wo - 2501004489 part no : m4d154810zm ( pcb ) part still udr part requested - didn't link to sap pls assist to link as customer request urgent repair / to close job sheet | cat : application / software | delta _ hours : 3266. 45",
        'effect : dear team error occurred while processing the edi transaction interface subsidiary no subsidiary api name pana - pagitp - mgmt - eapi flow direction inbound source system pagitp end system sap s4hana file name no file name storage path no attachment error source mulesoft transaction id 946bd1f9 - c3a8 - 4309 - 9216 - df00e3efc6c4 error summary 500 source _ response _ send error details client connection was closed comments note : this is an automated mail, please do not reply. thanks and regards, apac support, panasonic | cat : application / software | delta _ hours : 2826. 65',
        'effect : dear team error occurred while processing the edi transaction interface change - over subsidiary pau api name pana - sdesk - ext - eapi flow direction outbound source system pau end system zoho file name no file name storage path no attachment error source pana - sdesk - ext - eapi transaction id 819e3530 - a313 - 11ef - 8b0c - 02b4130d4440 error summary 500 expression error details expression : " org. mule. runtime. api. exception. muleruntimeexception - exception was found trying to retrieve the contents of file / zoho / live / outbound / status / remote / co _ status _ 20241115163317536699. csv org. mule. runtime. api. exception. muleruntimeexception : exception was found trying to retrieve the contents of file / zoho / live / outbound / status / remote / co _ status _ 20241115163317536699. csv caused by : sftp error ( ssh _ fx _ failure ) : failure. at org. apache. sshd. sftp. client. impl. abstractsftpclient. throwstatusexception ( abstractsftpclient. java : 277 ) at org. apache. sshd. sftp. client. impl. abstractsftpclient. checkhandleresponse ( abstractsftpclient. java : 299 ) at org. apache. sshd. sftp. client. impl. abstractsftpclient. checkhandle ( abstractsftpclient. java : 290 ) at org. apache. sshd. sftp. client. impl. abstractsftpclient. open ( abstractsftpclient. java : 589 ) at org. apache. sshd. sftp. client. impl. sftpinputstreamasync. ( sftpinputstreamasync. java : 75 ) at org. apache. sshd. sftp. client. impl. abstractsftpclient. read ( abstractsftpclient. java : 1196 ) at org. apache. sshd. sftp. client. sftpclient. read ( sftpclient. java : 909 )',
        'effect : abap logic inquiry for the rr interface | cat : application / software | delta _ hours : 719. 37',
        'effect : dear team error occurred while processing the edi transaction interface change - over subsidiary pau api name pana - sdesk - ext - eapi flow direction outbound source system pau end system zoho file name no file name storage path no attachment error source pana - sdesk - ext - eapi transaction id 56fcce20 - 39ce - 11f0 - 8032 - eee27e3a0f38 error summary 500 expression error details " expression : \\ " org. mule. runtime. api. exception. muleruntimeexception - could not obtain connection to fetch file / zoho / live / outbound / status / remote / co _ status _ 20250526110806711817. csv \\ norg. mule. runtime. api. exception. muleruntimeexception : could not obtain connection to fetch file / zoho / live / outbound / status / remote / co _ status _ 20250526110806711817. csv \\ ncaused by : org. mule. runtime. api. connection. connectionexception : could not establish sftp connection with host :\'10. 86. 48. 62\'at port :\'22\'- session. connect : java. net. sockettimeoutexception : read timed out \\ ncaused by : com. jcraft. jsch. jschexception : session. connect : java. net. sockettimeoutexception : read timed out \\ n \\ tat com. jcraft. jsch. session. connect ( session. java : 565 ) \\ n \\ tat com. jcraft. jsch. session. connect ( session. java : 183 ) \\ n \\ tat org. mule. extension. sftp. internal. connection. sftpclient. connect ( sftpclient. java : 182 ) \\ n \\ tat org. mule. extension. sftp. internal. connection. sftpclient. login ( sftpclient. java : 164 ) \\ n \\ tat org. mule. extension. sftp. internal. connection. sftpconnectionprovider. connect ( sftpconnectionprovider. java : 141 ) \\ n \\ tat',
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

* Size: 2,737 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                        | sentence_1                                                                                        | label                                                          |
  |:--------|:--------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                            | string                                                                                            | float                                                          |
  | details | <ul><li>min: 31 characters</li><li>mean: 362.65 characters</li><li>max: 1613 characters</li></ul> | <ul><li>min: 68 characters</li><li>mean: 530.45 characters</li><li>max: 1661 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.26</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                          | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>cause : pgi - so - 2505003793 is appear in sfdc & partner portal but not appear ( cannot be read in sap ), please fix. \| cat : application / software</code> | <code>effect : job : pm - wo - 2501004489 part no : m4d154810zm ( pcb ) part still udr part requested - didn't link to sap pls assist to link as customer request urgent repair / to close job sheet \| cat : application / software \| delta _ hours : 3266. 45</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | <code>0.0</code> |
  | <code>cause : pls cancel ews psv - ews - 2503000221 in sfdc and don't trigger to sap \| cat : application / software</code>                                         | <code>effect : dear team error occurred while processing the edi transaction interface subsidiary no subsidiary api name pana - pagitp - mgmt - eapi flow direction inbound source system pagitp end system sap s4hana file name no file name storage path no attachment error source mulesoft transaction id 946bd1f9 - c3a8 - 4309 - 9216 - df00e3efc6c4 error summary 500 source _ response _ send error details client connection was closed comments note : this is an automated mail, please do not reply. thanks and regards, apac support, panasonic \| cat : application / software \| delta _ hours : 2826. 65</code>                                                                                                                                                                                                                                                                                                                                                                                                                          | <code>0.0</code> |
  | <code>cause : material : srf3704 - kd10 \| cat : application / software</code>                                                                                      | <code>effect : dear team error occurred while processing the edi transaction interface change - over subsidiary pau api name pana - sdesk - ext - eapi flow direction outbound source system pau end system zoho file name no file name storage path no attachment error source pana - sdesk - ext - eapi transaction id 819e3530 - a313 - 11ef - 8b0c - 02b4130d4440 error summary 500 expression error details expression : " org. mule. runtime. api. exception. muleruntimeexception - exception was found trying to retrieve the contents of file / zoho / live / outbound / status / remote / co _ status _ 20241115163317536699. csv org. mule. runtime. api. exception. muleruntimeexception : exception was found trying to retrieve the contents of file / zoho / live / outbound / status / remote / co _ status _ 20241115163317536699. csv caused by : sftp error ( ssh _ fx _ failure ) : failure. at org. apache. sshd. sftp. client. impl. abstractsftpclient. throwstatusexception ( abstractsftpclient. java : 277 ) at org....</code> | <code>0.0</code> |
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

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `eval_strategy`: no
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Framework Versions
- Python: 3.11.14
- Sentence Transformers: 5.2.2
- Transformers: 5.2.0
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