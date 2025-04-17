### Unsloth Adaptation Cheat Sheet (Custom Model + HF Trainer)

**Goal:** Use Unsloth's optimizations (`FastModel`, 4-bit QLoRA) for memory efficiency and speed, while training a custom model architecture (wrapper + emotion layer + custom forward pass) using the standard `transformers.Trainer`.

**Core Strategy:**

1.  Load the base model efficiently with Unsloth's `FastModel`.
2.  Wrap the `FastModel` instance in a custom `nn.Module` that adds your logic.
3.  Apply PEFT (LoRA) to the wrapper.
4.  Prepare data with a custom `DataCollator` that passes your extra inputs (`emotion_vector`).
5.  Use the standard `transformers.Trainer`, configured to handle the custom data flow.

**Key Issues & Solutions:**
