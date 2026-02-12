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
- source_sentence: khanh nguyen/shirish waragade, there is another 3 invoice user
    reported that did not interface to sap. can help to check? in d365 system, no
    error and shows posted successfully. sscem2405000002 sscem2405000003 av3407034
  sentences:
  - '#idoc 0000000071388204 error'
  - 'request change password in pnq sap id: 70p9265'
  - khanh nguyen/shirish waragade, there is another 3 invoice user reported that did
    not interface to sap. can help to check? in d365 system, no error and shows posted
    successfully. sscem2405000002 sscem2405000003 av3407034
- source_sentence: pnz complained that pnz sales data in papmap daily sales progress
    report is difference with their erp data. refer attached for more details and
    investigate. some categories also missing in papmap daily sales progress report.
  sentences:
  - pnz complained that pnz sales data in papmap daily sales progress report is difference
    with their erp data. refer attached for more details and investigate. some categories
    also missing in papmap daily sales progress report.
  - 'user: lilibeth caspe/ jian hui issue: edo-s1-24004332 - posting error in sap
    due to purchasing document 100027431 not yet released. now the purchase order
    already posted successful. need ams to rerun the job to repost.'
  - pacmy-sales order-2403003435 to have been confirm at sap but status at salesforce
    take time to change and make it more longer to billing
- source_sentence: 'user: michelle // ema-s1-24000376 servicenow: segment code 00809511970000038128
    / batch a0y5 sap: segment code 00809511970000038128 / batch a0aj. batch master
    a0aj missing customer code, unabled to add to update/save due to sales group code
    is mandatory field.'
  sentences:
  - 'user : caremen delivery order 551008915 block for delivery but user confirmed
    the credit limit granted is able to cover. bp : 80935820 help to check the credit
    exposure .'
  - b0dccf000002 vmi stock quantity should be 0 but it showed 125000 in api report.
    help to check the reason and how to correct it. this case is related to month-end
    closing, give it priority.
  - 'user: michelle // ema-s1-24000376 servicenow: segment code 00809511970000038128
    / batch a0y5 sap: segment code 00809511970000038128 / batch a0aj. batch master
    a0aj missing customer code, unabled to add to update/save due to sales group code
    is mandatory field.'
- source_sentence: 'help to check, should be 1 cn created, why 4 return send to sap
    and 3 cn already created. invoice: pcmsi240300064 cn : pcmsc240300001'
  sentences:
  - 'help to check, should be 1 cn created, why 4 return send to sap and 3 cn already
    created. invoice: pcmsi240300064 cn : pcmsc240300001'
  - pacmy-sales order-2403003074 have been confirm to sap but status at salesforce
    take time to change, and make it longer time to billing
  - abap error still prompt in vl32n and zmsd_do01 for 551008965.
- source_sentence: production settlement error at this moment, all our balance inventories
    has been shipped out and updated in the sap last week. all the inventories physically
    already zero in our factory but in the system, it still show the balance rm 104.02
    in inventory due to 29 orders as per showed in the below screenshot.
  sentences:
  - issue to clear cogi in plant iaf3 plant for kz-hips
  - production settlement error at this moment, all our balance inventories has been
    shipped out and updated in the sap last week. all the inventories physically already
    zero in our factory but in the system, it still show the balance rm 104.02 in
    inventory due to 29 orders as per showed in the below screenshot.
  - we found daily shipment plan report (ztgasd_0018) show etd bkk by sea incorrect
    date detail as attached file.
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
    'production settlement error at this moment, all our balance inventories has been shipped out and updated in the sap last week. all the inventories physically already zero in our factory but in the system, it still show the balance rm 104.02 in inventory due to 29 orders as per showed in the below screenshot.',
    'production settlement error at this moment, all our balance inventories has been shipped out and updated in the sap last week. all the inventories physically already zero in our factory but in the system, it still show the balance rm 104.02 in inventory due to 29 orders as per showed in the below screenshot.',
    'we found daily shipment plan report (ztgasd_0018) show etd bkk by sea incorrect date detail as attached file.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 1.0000, 0.2100],
#         [1.0000, 1.0000, 0.2100],
#         [0.2100, 0.2100, 1.0000]])
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
  | details | <ul><li>min: 5 tokens</li><li>mean: 49.75 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 49.75 tokens</li><li>max: 128 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>pm-wo-2207000157_delivered but cannot consume, part number: axw2331+a30a0</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | <code>pm-wo-2207000157_delivered but cannot consume, part number: axw2331+a30a0</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
  | <code>reverse tdd variance in diferent period</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | <code>reverse tdd variance in diferent period</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
  | <code>shruti, i tried to debug this, but the steps on search are never captured in the debug logs users are having issues when they need to add a product consumption in the wo. is there anything wrong with the way we add the filter criteria? [cid:image001.png@01da485f.9d71e0f0] see the video attached. it looks to me that salesforce might be running too many lines until it finds the work order line item? this is also happening when people are trying to find ship to address upon creation of a wo under an account, but i could not find an example to capture... happy to have a call to show you marina soares the information contained in this message is privileged and intended only for the recipients named. if the reader is not a representative of the intended recipient, any review, dissemination or copying of this message or the information it contains is prohibited. if you have received this message in error, immediately notify the sender, and delete the original message and attachments.</code> | <code>shruti, i tried to debug this, but the steps on search are never captured in the debug logs users are having issues when they need to add a product consumption in the wo. is there anything wrong with the way we add the filter criteria? [cid:image001.png@01da485f.9d71e0f0] see the video attached. it looks to me that salesforce might be running too many lines until it finds the work order line item? this is also happening when people are trying to find ship to address upon creation of a wo under an account, but i could not find an example to capture... happy to have a call to show you marina soares the information contained in this message is privileged and intended only for the recipients named. if the reader is not a representative of the intended recipient, any review, dissemination or copying of this message or the information it contains is prohibited. if you have received this message in error, immediately notify the sender, and delete the original message and attachments.</code> |
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
- `num_train_epochs`: 2
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
- `num_train_epochs`: 2
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