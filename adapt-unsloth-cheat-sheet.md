### Unsloth Adaptation Cheat Sheet (Custom Model + HF Trainer)

**Goal:** Use Unsloth's optimizations (`FastModel`, 4-bit QLoRA) for memory efficiency and speed, while training a custom model architecture (wrapper + emotion layer + custom forward pass) using the standard `transformers.Trainer`.

**Core Strategy:**

1.  Load the base model "efficiently" with Unsloth's `FastModel`.
2.  Wrap the `FastModel` instance in a custom `nn.Module` with custom logic.
3.  Apply PEFT (LoRA) to the wrapper.
4.  Prepare data with a custom `DataCollator` that passes your extra inputs (`emotion_vector`).
5.  Use the standard `transformers.Trainer`, configured to handle the custom data flow.

**Key Issues & Solutions:**

1.  **Issue:** Unsloth doesn't always provide helper functions (like `get_peft_model`) as top-level imports; they are methods or integrated differently.
    *   **Hint:** Use standard libraries (`peft.get_peft_model`, `peft.LoraConfig`) after loading the model with `unsloth.FastModel.from_pretrained()`.

2.  **Issue:** `AttributeError` on standard config attributes (`hidden_size`, `vocab_size`) when accessing `model.config`. Unsloth's loaded model might have a different config structure.
    *   **Hint:** The configuration attributes for Gemma models loaded via Unsloth often reside within a nested `text_config` attribute.
    *   **Solution:** Use `try-except` blocks to robustly access attributes like `model.config.text_size.hidden_size` or `model.config.text_config.vocab_size`. Print `dir(model.config)` and `dir(model.config.text_config)` if needed for debugging, or take config from original model I think we stick to this.

3.  **Issue:** `KeyError: 'emotion_vectors'` in the `DataCollator`. The custom column is missing from the batch.
    *   **Hint:** The `transformers.Trainer` by default removes dataset columns that are not standard model inputs to save memory/prevent errors.
    *   **Solution:** Set `remove_unused_columns=False` in `TrainingArguments`. This ensures the custom data column reaches custom collator.

4.  **Issue:** Length mismatches (`AssertionError`) in the `DataCollator` when trying to pad/truncate lists.
    *   **Hint:** Data preparation should be done consistently *before* the collator. The collator's job should be primarily converting pre-sized lists/dictionaries into tensors and handling label masking.
    *   **Solution:** ensure *each processed sample's* lists (`input_ids`, `attention_mask`, `emotion_vectors`) are padded or truncated to the exact target `max_length` *before* being returned. The `DataCollator` then expects inputs of this fixed length.

5.  **Issue:** Standard `Trainer` compatibility with a wrapped model that overrides `forward`.
    *   **Hint:** The `Trainer` expects certain attributes (like `config`, `device`, `dtype`, `gradient_checkpointing_enable`, and potentially `base_model` if using PEFT) to be accessible on the model object passed to it.
    *   **Solution:** Implement a `__getattr__` method in wrapper class (`EmotionalGemmaUnslothWrapper`) to delegate attribute access to the underlying Unsloth `FastModel` instance if the attribute is not found directly on the wrapper. This makes the wrapper behave more like a standard HF model for the Trainer. Implement specific delegation methods (like `get_input_embeddings`, `set_input_embeddings`, `gradient_checkpointing_enable`) if `__getattr__` isn't sufficient or if specific logic is needed.

**Data Pipeline Hints:**

*   **Dataset Script (`dataset.py`):** Responsible for loading raw data, applying chat template, parsing custom tags (emotions), tokenizing segments, combining tokens/vectors/mask, and crucially, padding/truncating *each sample* to the exact `max_length`. Output should be a list of dictionaries, where each dictionary represents one sample with keys like `input_ids`, `attention_mask`, `emotion_vectors`, and lists of size `max_length`.
*   **Data Collator (`DataCollatorForEmotionalLlama`):** Responsible for taking a list of dictionaries (the batch) from the Dataset/DataLoader (which are already sized to `max_length` thanks to `dataset.py`), converting the lists into PyTorch tensors, adding the `labels` key with appropriate -100 masking (masking padding and non-target tokens like user turns/prompts). It receives the `"emotion_vectors"` list and outputs the `"emotion_vector"` tensor in the batch dictionary.
*   **Model (`EmotionalGemmaUnslothWrapper`):** The `forward` method expects `input_ids`, `attention_mask`, `labels`, and `emotion_vector` (singular name from collator batch). It uses `input_ids` (or computes `inputs_embeds`), modifies `inputs_embeds` using `emotion_vector`, passes `inputs_embeds` to the wrapped `self.model`, and computes the loss using the output `logits` and the `labels`.

**Model & Training Specifics:**

*   **Wrapper:** Inherit from `nn.Module` and hold the `FastModel` as an attribute (`self.model`).
*   **Emotion Layer:** `emotion_proj_embed` is added as a standard `nn.Sequential` within the wrapper.
*   **PEFT:** Apply `peft.get_peft_model` to the *wrapper instance*.
*   **Trainable Parameters:** Manually set `param.requires_grad = True` for emotion projection layer's parameters *after*😔 applying PEFT, as PEFT might freeze them otherwise. `model.print_trainable_parameters()` to verify.
*   **Optimizer:** Use optimizer groups to set different learning rates for LoRA adapters and your custom emotion layer (often a higher LR for the small custom layer, with weight decay only on LoRA/base if applicable).
*   **Gradient Checkpointing:** Optional, `TrainingArguments` (`gradient_checkpointing=True`, `gradient_checkpointing_kwargs={"use_reentrant": False}`).
*   **Saving:** Save PEFT adapters using `model.save_pretrained()`. Manually save the state dictionary of emotional proj layer (`model_wrapper.emotion_proj_embed.state_dict()`) to a separate file (`.pth`). Access the layer via `final_model.base_model.emotion_proj_embed` if PEFT is applied. *Remember to move tensors to CPU before saving state dicts if needed.*
