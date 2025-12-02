# Notes

Yes, there are several other loss functions you can use. The choice depends on the format of your training data and your specific goal. The one used in the notebook, MultipleNegativesRankingLoss, is very effective, but here are some powerful alternatives available in the sentence-transformers library:

1. `ContrastiveLoss`

   - How it Works: Takes a pair of sentences. If the pair is positive (similar), it pushes their embeddings together. If the pair is negative (dissimilar), it pushes them apart until they are separated by a minimum distance (a margin).
   - Best For: When you have a dataset of explicit positive and negative pairs. It's a classic and solid choice for Siamese networks.

2. `TripletLoss`

   - How it Works: This loss requires a "triplet": an anchor sentence, a positive sentence (similar to the anchor), and a negative sentence (dissimilar). It trains the model to ensure the distance between the anchor and positive is smaller than the distance between the anchor and negative by at least a certain margin.
   - Best For: Learning a fine-grained ranking. It directly optimizes for the question, "Is ticket A more similar to B than it is to C?" This can be very powerful but requires you to create explicit triplets of data, which can be complex.

3. `CosineSimilarityLoss`
   - How it Works: A simple and intuitive loss. It tries to make the model's cosine similarity score for a pair of sentences as close as possible to a true label (a score from 0.0 to 1.0).
   - Best For: When your dataset has gold-standard similarity scores (e.g., human-annotated scores like 0.8, 0.5, etc.), not just binary "similar/dissimilar" labels.

Why was `MultipleNegativesRankingLoss` used? It's extremely efficient. For a batch of 16 positive pairs, it automatically creates thousands of negative pairs by treating all non-matching combinations within the batch as negatives. This provides a very strong training signal from a relatively small amount of labeled data.

More Efficient Ways of Training

"Efficiency" can mean faster training, using less memory, or getting better results from less data. Here are several ways to improve it:

Faster Training & Less Memory

1.  Mixed Precision Training: This is one of the most effective methods. It uses a mix of 16-bit and 32-bit floating-point numbers during training. On modern GPUs, this can dramatically speed up training (30-50% faster) and reduce memory usage by nearly half with a negligible impact on accuracy.

    - Implementation: This is often a simple boolean flag in the training library (e.g., in transformers.Trainer).

2.  Parameter-Efficient Fine-Tuning (PEFT / LoRA): This is a state-of-the-art technique. Instead of fine-tuning all millions of parameters in the model, you freeze the base model and only train a tiny set of new "adapter" layers (a technique called Low-Rank Adaptation or LoRA).

    - Benefits: Drastically reduces the memory required for training, allowing you to fine-tune much larger models on the same hardware. Training is also much faster.

3.  Use a Smaller Model: The simplest method. Fine-tuning a model like all-MiniLM-L6-v2 will be significantly faster and require less memory than all-mpnet-base-v2.

More Data-Efficient Training

1.  Data Augmentation: The notebook already uses a simple form of this. You could use more advanced techniques to create high-quality synthetic training data:

    - Back-Translation: Translate your text to another language and then back to English to get a paraphrase.
    - Synonym Replacement: Use a thesaurus like WordNet to replace words with their synonyms.
    - Generative Models: Use a model like GPT to generate paraphrases of your tickets.

2.  Active Learning: Instead of randomly labeling data, you can build a system where the model points out the unlabeled examples it is most "confused" about. You then label only those highly informative examples, allowing the model to learn much faster with fewer labels. This is a more complex but very powerful workflow.

Of course. Let's focus on maximizing the model's score using only the data you already have (dummy_data_promax.csv and relationship_pairs.json).

Here are the most effective ways to improve your model's performance, ordered from highest to lowest expected impact.

1. Fix the Learning Rate Bug (Critical First Step)

As noted in findings.md, the learning rate you set in CONFIG is currently not being used. The model is training with the library's default. This is the first and most important thing to fix.

In the model.fit() call, you need to pass the learning rate via optimizer_params.

Change this:
1 model.fit(
2 train_objectives=[(train_dataloader, train_loss)],
3 # ... other params
4 )

To this:

1 model.fit(
2 train_objectives=[(train_dataloader, train_loss)],
3 epochs=CONFIG['epochs'],
4 warmup_steps=CONFIG['warmup_steps'],
5 optimizer_params={'lr': CONFIG['learning_rate']}, # <-- ADD THIS LINE
6 # ... other params
7 )
Fine-tuning often requires a small learning rate. A good starting point is between 1e-5 and 5e-5.

2. Improve Positive and Negative Pair Generation

This is the single most impactful area for improvement. The model is only as good as the examples it learns from.

- Create Higher-Quality Positive Pairs: Instead of just grouping by Category and Subcategory, you can be more selective. Create a rule that a positive pair must also share some keyword overlap in their short descriptions. This ensures the pairs are not just topically related but textually similar, providing a stronger training signal.
- Create Better "Hard" Negatives: The current strategy finds negatives with some keyword overlap. To make them even "harder" and more informative, find pairs that a weaker, pre-trained model (or a keyword-based search like BM25) thinks are highly similar but which you know are not related (i.e., they have different categories). This forces your model to learn the
  subtle differences that traditional methods miss.

  3. Use More Advanced Data Augmentation

  The current augmentation (word shuffling/dropping) is basic. You can create much more realistic and diverse training examples using a library like nlpaug.

  First, install it: pip install nlpaug

  Then, you can replace the simple augment_text_simple function with something more powerful, like synonym replacement.

  Example:

  1 import nlpaug.augmenter.word as naw
  2
  3 # You only need to create this once
  4 aug = naw.SynonymAug(aug_src='wordnet')
  5
  6 def augment_text_advanced(text):
  7 # This will replace some words with their synonyms
  8 return aug.augment(text)
  9
  10 # Then use this function when creating your augmented pairs
  11 augmented_pairs.append({
  12 # ...
  13 "text1": augment_text_advanced(pair["text1"]),
  14 # ...
  15 })
  This creates more varied and semantically rich examples for the model to learn from.

  4. Experiment with a Different Loss Function

  While MultipleNegativesRankingLoss is excellent and efficient, `TripletLoss` can sometimes yield better results by being more explicit.

- How to Implement: 1. You would need to change your data loading process to create explicit "triplets": (anchor, positive, negative). 2. An anchor is a ticket. A positive is a known similar ticket. A negative is a known dissimilar ticket (ideally a "hard negative"). 3. You would then switch the loss function: train_loss = losses.TripletLoss(model=model) 4. The data loader would feed these triplets into the model during training.

  This approach can be more powerful if your hard negatives are well-chosen.

  5. Tune max_seq_length

  The current sequence length is 256 tokens. I recommend analyzing your data: are your ticket descriptions frequently longer than that? If so, important context is being cut off.

- Action: Calculate the token length for all your ticket descriptions. If a significant portion (e.g., >15%) are longer than 256, consider increasing max_seq_length to 384 or 512.
- Trade-off: Be aware that this will increase memory consumption and significantly slow down training, so try the other methods first.
