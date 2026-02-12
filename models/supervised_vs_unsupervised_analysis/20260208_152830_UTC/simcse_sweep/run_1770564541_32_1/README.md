---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:960
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: 'request change password in pnq sap id: 70p9265'
  sentences:
  - 'request change password in pnq sap id: 70p9265'
  - 'regarding planed order interface from aipas to sap. it should be deleted old
    planed order by sap not delete it. we would like you to replicate in tgq and reconfirm
    the correct program logic. sample planed order: 3411000669 from tgp900.'
  - 'check the sales order - psv-sales order-2404011644: we don''t see the e-invoice
    in salesforce for this sales order'
- source_sentence: case no 4. missing invoice no!!! urgent!!! urgent!!! urgent!!!
  sentences:
  - case no 4. missing invoice no!!! urgent!!! urgent!!! urgent!!!
  - check as one of the fields may be missing or invalid in the raw data. if unsure
    what is the correct value, ask advice from user.
  - cz-rtc4a in mmbe not able to display sales order/customer details
- source_sentence: '"although there is information about warranty expiry date in warranty
    registration, it is not updated on asset''s warranty expiry date. check the file
    and fix the expiry date of these assets. by the way, this issue happens frequently,
    find out the root cause and permanent solutions to prevent its happening in the
    future. "'
  sentences:
  - pacmy-sales order-2404003189
  - rashi gururani is unable to print reports in sap pnp.
  - '"although there is information about warranty expiry date in warranty registration,
    it is not updated on asset''s warranty expiry date. check the file and fix the
    expiry date of these assets. by the way, this issue happens frequently, find out
    the root cause and permanent solutions to prevent its happening in the future.
    "'
- source_sentence: 'help for this issue : if we check, part with detail bellow : no
    sales order : pgi-sales order-2402001873 no wo : pgi-wo-2402002421 part : th-65mx650g-rs
    is appear in partner portal (nsc), the status is “new” but if we check in sales
    force, this part is not appear , and we check this part is not appear in sap genesis
    too, so, help to appear this part in sales force and sap genesis and we can continue
    the next process note : for detail see my email & itmaas (servicenow) attachment
    too,'
  sentences:
  - 'help for this issue : if we check, part with detail bellow : no sales order :
    pgi-sales order-2402001873 no wo : pgi-wo-2402002421 part : th-65mx650g-rs is
    appear in partner portal (nsc), the status is “new” but if we check in sales force,
    this part is not appear , and we check this part is not appear in sap genesis
    too, so, help to appear this part in sales force and sap genesis and we can continue
    the next process note : for detail see my email & itmaas (servicenow) attachment
    too,'
  - we have been noticed cs thai ha 2 account several times displayed as ""cs thai
    ha 2 m?i thái hà"" which is in completely wrong form. support us to correct it
    to fixed name ""cs thai ha 2""
  - 'error msg: valuation data for material tripartite is locked by the user ediuser500
    help to repost bala has assisted, assign ticket to him'
- source_sentence: document number 9536005923 9536005937 9536005944 9536006439 9536006612
    9537500636 9536006587 9536006588 9536006244 9536006324 9536006340 9536006368 9536006385
    9536006395 9536006414 9536006430
  sentences:
  - 'team, expected : when a non-project work order is created it creates a sales
    order automatically. issue: when non-project work orders are created some of them
    are not leading to the creation of a related sales order. example: wo 00072954
    (https://hussmannnz.lightning.force.computer/lightning/r/workorder/0wo5j0000002tkmgam/view)
    this work order has no sales order created. i am attaching an example of this
    issue with this ticket'
  - pnz-asc price book to harrison ac-sap 5000026818 team panda sap account 5000026891
  - document number 9536005923 9536005937 9536005944 9536006439 9536006612 9537500636
    9536006587 9536006588 9536006244 9536006324 9536006340 9536006368 9536006385 9536006395
    9536006414 9536006430
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
    'document number 9536005923 9536005937 9536005944 9536006439 9536006612 9537500636 9536006587 9536006588 9536006244 9536006324 9536006340 9536006368 9536006385 9536006395 9536006414 9536006430',
    'document number 9536005923 9536005937 9536005944 9536006439 9536006612 9537500636 9536006587 9536006588 9536006244 9536006324 9536006340 9536006368 9536006385 9536006395 9536006414 9536006430',
    'pnz-asc price book to harrison ac-sap 5000026818 team panda sap account 5000026891',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 1.0000, 0.2495],
#         [1.0000, 1.0000, 0.2495],
#         [0.2495, 0.2495, 1.0000]])
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

* Size: 960 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 960 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             |
  | details | <ul><li>min: 5 tokens</li><li>mean: 49.76 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 49.76 tokens</li><li>max: 128 tokens</li></ul> |
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

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 1
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
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