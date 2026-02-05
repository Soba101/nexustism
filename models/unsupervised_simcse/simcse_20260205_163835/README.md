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
- source_sentence: claim-000003591 status changed to submitted in salesforce and triggered
    to sap on 30/04/2024, 1:46 pm. check if it is existing in sap
  sentences:
  - claim-000003591 status changed to submitted in salesforce and triggered to sap
    on 30/04/2024, 1:46 pm. check if it is existing in sap
  - pacmy-sales order-2404003292
  - 'model: wv-s32302-f2l1'
- source_sentence: account 400100 require assignment to co object error prompted while
    user post variance.
  sentences:
  - account 400100 require assignment to co object error prompted while user post
    variance.
  - 'req0012859/ritm0012904 - setup pasap factory calendar p3 @ujwal chaudhari san,
    may i confirmed as below sr.no months-year current factory calendar (p3) holiday
    dates new request to update factory calander (p3) holiday dates 1 april -2024
    13,14,15 add holiday: 11,12,13,14,15,16 2 may -2024 1 1 (no changes) 3 jun -2024
    no holiday no changes 4 july -2024 no holiday no changes 5 aug -2024 12 remove
    holiday: 12 6 sep -2024 no holiday no changes 7 oct -2024 23 remove holiday: 23
    8 nov -2024 no holiday no changes 9 dec -2024 5,31 remove holiday: 5 add holiday:
    28,29,30,31 10 jan -2025 1 add holiday: 1,2,3,4,5 11 feb -2025 no holiday no changes
    12 mar -2025 no holiday no changes'
  - 'user: chen yu poh issue: unable to rectify load summary error. advise what of
    error based on the msg per screenshot.'
- source_sentence: urgent - psp-wo-2405000473 - the service charge should be s$160
    instead of s$100.
  sentences:
  - '1 product has 2 assets and 2 warranty registration. when delete 1 of them, it
    will be cancelled at e-warranty; then comeback to another asset to edit any field
    & save. 1. serial: 2191290023 model: na-v90fx2lvt 2. serial: 21n2101020 model:
    na-v90fx1lvt server error : 500 : {"access_token":"n6bc31h-iee_wvyowqhg2yklitxyfgtqhvqokjb_yrxpsb3dtpw89sgnavrmkzok6egnnrktpzhanpslpgmv0w2xca4t6a6ivsgawiiawriynlxiyvpy62ovckgfdrclrase0chui4c388rbgari-qlirkogwhel2yjq5wtiscqclmzzxwwxm_suy7btgqpwkir_-g5-blaxbcmrwtwrbvfc-yeex6vci1gdhg-exvpgegv9eqpzkpf1v9ecomegubslknurruplpbq68smfsfubbfh9jxmkssesgjwfc0pwhqq4ti_dctkh0odp-9qityngpjnwbz9otilkskm9egpa4tfrbr1mwjyqwhkwxydbah5jj9x8nl-fdromqrdtljtddbbg8lme0rhlyw40xoowcqa","token_type":"bearer","expires_in":129599,"appuserid":"9","tenantuserid":"13","tenantuserguid":"bf6b80b1-bfd7-4b54-83d7-a2f9da361a36","requireotp":"false","status":"ok","statusmsg":"success",".issued":"thu,
    28 mar 2024 10:10:53 gmt",".expires":"fri, 29 mar 2024 22:10:53 gmt"}'
  - support, advise any changes done from your end because all the attachments cannot
    be downloaded from eworkplace. it showed file can’t be downloaded securely for
    all the attachment. https://www.eworkplace.intra.panasonic.computer.sg/sites/pfsap/sitepages/apachome.aspx
    --> download with error https://eworkplace.apac.panasonic.computer/sites/pfsap/sitepages/apachome.aspx?
    --> download without error
  - urgent - psp-wo-2405000473 - the service charge should be s$160 instead of s$100.
- source_sentence: 'change quantity on hand and quantity on available with detail
    bellow : current : locations : kediri product item number : pi-187574 product
    name : w0401-42504 bin : a10019 quantity on hand : 0,00 quantity on available
    : 1,00 quantity unit of : computer should be (please change) : quantity on hand
    : 0,00 quantity on available : 0,00 and we wait the confirmation'
  sentences:
  - 'change quantity on hand and quantity on available with detail bellow : current
    : locations : kediri product item number : pi-187574 product name : w0401-42504
    bin : a10019 quantity on hand : 0,00 quantity on available : 1,00 quantity unit
    of : computer should be (please change) : quantity on hand : 0,00 quantity on
    available : 0,00 and we wait the confirmation'
  - 'update: for inc0015878 & inc0015794 is now under care of k poorna chandra reddy
    for generating corrected files as a fix to this issue. balakumar ganesan, k poorna
    chandra reddy i have added the work notes to both the tickets for your reference
    to proceed accordingly.'
  - issue raised by francy teh help to check why ecndn posting fail. error message
    in sap is not clear.
- source_sentence: no applicable approval process was found. refer to image attached.
  sentences:
  - help to check why no output to trigger the 2 docs to servicenow 5100045347 , 5100045327
  - check warranty registration pdf of w-5637633. expiry date warranty sale 35 months
    is wrong (correct 36months)
  - no applicable approval process was found. refer to image attached.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision e8c3b32edf5434bc2275fc9bab85f82640a19130 -->
- **Maximum Sequence Length:** 384 tokens
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
  (0): Transformer({'max_seq_length': 384, 'do_lower_case': False, 'architecture': 'MPNetModel'})
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
    'no applicable approval process was found. refer to image attached.',
    'no applicable approval process was found. refer to image attached.',
    'check warranty registration pdf of w-5637633. expiry date warranty sale 35 months is wrong (correct 36months)',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 1.0000, 0.0643],
#         [1.0000, 1.0000, 0.0643],
#         [0.0643, 0.0643, 1.0000]])
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
  | details | <ul><li>min: 6 tokens</li><li>mean: 56.19 tokens</li><li>max: 384 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 56.19 tokens</li><li>max: 384 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                         | sentence_1                                                                                                                                                                                                                                                         |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>user informed this delivery order 3000791138 cannot outbound to sato. after checked the sales order 1000922983, already updated customer group 5. customer group 5 field already updated in customer master 00023150-1. could you check and feedback.</code> | <code>user informed this delivery order 3000791138 cannot outbound to sato. after checked the sales order 1000922983, already updated customer group 5. customer group 5 field already updated in customer master 00023150-1. could you check and feedback.</code> |
  | <code>settlement failed in orders, error refer to attachment. order number : 106538274</code>                                                                                                                                                                      | <code>settlement failed in orders, error refer to attachment. order number : 106538274</code>                                                                                                                                                                      |
  | <code>delivery order does not show reprint if printed for second time.</code>                                                                                                                                                                                      | <code>delivery order does not show reprint if printed for second time.</code>                                                                                                                                                                                      |
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

- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `num_train_epochs`: 1
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
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