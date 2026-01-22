---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:2792
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
    ["cause : dear team error occurred while processing the edi transaction. please find the details below and attached is the file associated to the transaction. interface gitp subsidiary pidsmy api name inds - global - if - mgmt - papi flow direction outbound source system sap end system ibmmq file name 0000000001763941 _ 007a05c0 - b21c - 11ef - 98df - 9601f4f6b381. xml storage path / inds / prod / outbound / nocompany / ordrsp / 0000000001763941 _ 007a05c0 - b21c - 11ef - 98df - 9601f4f6b381. xml error source mulesoft transaction id 007a05c0 - b21c - 11ef - 98df - 9601f4f6b381 error summary 500 connectivity error details * * * * * * * * * * * 443 / api / v1 / sd / order / confirmation'failed : remotely closed. comments unable to find 0000000001763941 _ 007a05c0 - b21c", "effect : dear team error occurred while processing the edi transaction. please find the details below and attached is the file associated to the transaction. interface gitp subsidiary pidsmy api name inds - global - if - mgmt - papi flow direction outbound source system sap end system ibmmq file name 0000000001769646 _ 040157c0 - b21c - 11ef - 98df - 9601f4f6b381. xml storage path / inds / prod / outbound / nocompany / ordrsp / 0000000001769646 _ 040157c0 - b21c - 11ef - 98df - 9601f4f6b381. xml error source mulesoft transaction id 040157c0 - b21c - 11ef - 98df - 9601f4f6b381 error summary 500 connectivity error details * * * * * * * * * * * 443 / api / v1 / aleaud'failed : remotely closed. comments unable to find 0000000001769646 _ 040157c0 - b21c - 11ef - 98df - 9601"],
    ["cause : authorized member can't access sap. | cat : application / software", 'effect : 3 ar pass data : 1 ). pgi so 2410000709 2 ). pgi so 2410000882 3 ). pgi so 2409012425 status is already paid, but not not included in the " gdn inv reg pass report buy part paid " report, please include all this data in the report, for detail please see the attchment, thanks | cat : application / software | delta _ hours : - 1921. 30'],
    ["cause : i can't edit my esanction even though finance azah san re - opened for topup. # iaf2015297 | cat : nan", 'effect : for wo : psv - wo - 2412009287 can not find charge : " phi v? n chuy? n... " to add into work order, pls check and fix it. | cat : application / software | delta _ hours : 766. 18'],
    ['cause : item part axw - 16038000a25466, that unable to be purchased ( po to factory ) we used ( tcode : me21n ), zeru san ( user ) already process dmr, pir already ok but only unable to po to factory, for detail we attached in servicenow, thanks | cat : application / software', 'effect : user : fenny issue : user did cnlexp under invoice 130023838 due to revise in qty. however the sales advise in the new billing ( 130024123 ) is failed as attached screenshot. | cat : application / software | delta _ hours : 23. 62'],
    ['cause : dear team error occurred while processing the edi transaction. please find the details below and attached is the file associated to the transaction. flow direction inbound subsidiary papvn - tl2 source system invoice end system sap file name na storage path api name sgst - fi - invoice - papi error source invoice transaction id 9793e450 - 3da7 - 11f0 - b397 - fe052cc11875 error summary < html > < head > < meta name = " viewport " content = " width = device - width, initial - scale = 1 " > < style type = " text / css " > / *! * bootstrap v3. 3. 5 ( http : / / getbootstrap. com ) * copyright 2011 - 2015 twitter, inc. * licensed under mit ( https : / / github. com / twbs / bootstrap / blob / master / license ) * / / *! normalize. css v3. 0. 3 | mit license | github. com / necolas / normalize. css * / html { font - family : sans - serif ; - ms - text - size - adjust : 100 % ; - webkit - text - size - adjust : 100 % ; } body { margin : 0 ; } h1 { font - size : 1. 7em ; font - weight : 400 ; line - height : 1. 3 ; margin : 0. 68em 0 ; } * { - webkit - box - sizing : border - box ; - moz - box - sizing : border - box ; box - sizing : border - box ; } * : before, * : after { - webkit - box - sizing : border - box ; - moz - box - sizing : border - box ; box - sizing : border - box ; } html { - webkit - tap - highlight - color : rgba ( 0, 0, 0, 0', 'effect : as i shared in msteams po, would like to inquire if we can check this bp code under papvn cause for pmpc we will be uploading this bp code 00039173 - 1 panasonic global excellence we wanted to check if there would be any issue if we proceed | cat : application / software | delta _ hours : - 6597. 73'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    "cause : dear team error occurred while processing the edi transaction. please find the details below and attached is the file associated to the transaction. interface gitp subsidiary pidsmy api name inds - global - if - mgmt - papi flow direction outbound source system sap end system ibmmq file name 0000000001763941 _ 007a05c0 - b21c - 11ef - 98df - 9601f4f6b381. xml storage path / inds / prod / outbound / nocompany / ordrsp / 0000000001763941 _ 007a05c0 - b21c - 11ef - 98df - 9601f4f6b381. xml error source mulesoft transaction id 007a05c0 - b21c - 11ef - 98df - 9601f4f6b381 error summary 500 connectivity error details * * * * * * * * * * * 443 / api / v1 / sd / order / confirmation'failed : remotely closed. comments unable to find 0000000001763941 _ 007a05c0 - b21c",
    [
        "effect : dear team error occurred while processing the edi transaction. please find the details below and attached is the file associated to the transaction. interface gitp subsidiary pidsmy api name inds - global - if - mgmt - papi flow direction outbound source system sap end system ibmmq file name 0000000001769646 _ 040157c0 - b21c - 11ef - 98df - 9601f4f6b381. xml storage path / inds / prod / outbound / nocompany / ordrsp / 0000000001769646 _ 040157c0 - b21c - 11ef - 98df - 9601f4f6b381. xml error source mulesoft transaction id 040157c0 - b21c - 11ef - 98df - 9601f4f6b381 error summary 500 connectivity error details * * * * * * * * * * * 443 / api / v1 / aleaud'failed : remotely closed. comments unable to find 0000000001769646 _ 040157c0 - b21c - 11ef - 98df - 9601",
        'effect : 3 ar pass data : 1 ). pgi so 2410000709 2 ). pgi so 2410000882 3 ). pgi so 2409012425 status is already paid, but not not included in the " gdn inv reg pass report buy part paid " report, please include all this data in the report, for detail please see the attchment, thanks | cat : application / software | delta _ hours : - 1921. 30',
        'effect : for wo : psv - wo - 2412009287 can not find charge : " phi v? n chuy? n... " to add into work order, pls check and fix it. | cat : application / software | delta _ hours : 766. 18',
        'effect : user : fenny issue : user did cnlexp under invoice 130023838 due to revise in qty. however the sales advise in the new billing ( 130024123 ) is failed as attached screenshot. | cat : application / software | delta _ hours : 23. 62',
        'effect : as i shared in msteams po, would like to inquire if we can check this bp code under papvn cause for pmpc we will be uploading this bp code 00039173 - 1 panasonic global excellence we wanted to check if there would be any issue if we proceed | cat : application / software | delta _ hours : - 6597. 73',
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

* Size: 2,792 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                        | sentence_1                                                                                        | label                                                          |
  |:--------|:--------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                            | string                                                                                            | float                                                          |
  | details | <ul><li>min: 43 characters</li><li>mean: 418.03 characters</li><li>max: 1529 characters</li></ul> | <ul><li>min: 71 characters</li><li>mean: 431.94 characters</li><li>max: 1570 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.23</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>cause : dear team error occurred while processing the edi transaction. please find the details below and attached is the file associated to the transaction. interface gitp subsidiary pidsmy api name inds - global - if - mgmt - papi flow direction outbound source system sap end system ibmmq file name 0000000001763941 _ 007a05c0 - b21c - 11ef - 98df - 9601f4f6b381. xml storage path / inds / prod / outbound / nocompany / ordrsp / 0000000001763941 _ 007a05c0 - b21c - 11ef - 98df - 9601f4f6b381. xml error source mulesoft transaction id 007a05c0 - b21c - 11ef - 98df - 9601f4f6b381 error summary 500 connectivity error details * * * * * * * * * * * 443 / api / v1 / sd / order / confirmation'failed : remotely closed. comments unable to find 0000000001763941 _ 007a05c0 - b21c</code> | <code>effect : dear team error occurred while processing the edi transaction. please find the details below and attached is the file associated to the transaction. interface gitp subsidiary pidsmy api name inds - global - if - mgmt - papi flow direction outbound source system sap end system ibmmq file name 0000000001769646 _ 040157c0 - b21c - 11ef - 98df - 9601f4f6b381. xml storage path / inds / prod / outbound / nocompany / ordrsp / 0000000001769646 _ 040157c0 - b21c - 11ef - 98df - 9601f4f6b381. xml error source mulesoft transaction id 040157c0 - b21c - 11ef - 98df - 9601f4f6b381 error summary 500 connectivity error details * * * * * * * * * * * 443 / api / v1 / aleaud'failed : remotely closed. comments unable to find 0000000001769646 _ 040157c0 - b21c - 11ef - 98df - 9601</code> | <code>1.0</code> |
  | <code>cause : authorized member can't access sap. \| cat : application / software</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | <code>effect : 3 ar pass data : 1 ). pgi so 2410000709 2 ). pgi so 2410000882 3 ). pgi so 2409012425 status is already paid, but not not included in the " gdn inv reg pass report buy part paid " report, please include all this data in the report, for detail please see the attchment, thanks \| cat : application / software \| delta _ hours : - 1921. 30</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                  | <code>0.0</code> |
  | <code>cause : i can't edit my esanction even though finance azah san re - opened for topup. # iaf2015297 \| cat : nan</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | <code>effect : for wo : psv - wo - 2412009287 can not find charge : " phi v? n chuy? n... " to add into work order, pls check and fix it. \| cat : application / software \| delta _ hours : 766. 18</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | <code>0.0</code> |
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
| 2.8571 | 500  | 0.4701        |


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