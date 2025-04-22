# train.py
# YOUR dataset.py should have create_huggingface_dataset() function like this:
# from dataset import create_huggingface_dataset
import os
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict
import json # Import json for dataset loading example

# Import the EmotionalLlamaModel and constants from the emotional_gemma.py file
from emotional_gemma import EmotionalLlamaModel, EMOTION_DIMENSIONS, EMOTION_DIMENSIONS_REFERENCE, MODEL_NAME



# Define the DataCollator for handling padding and adding emotion vectors
@dataclass
class DataCollatorForEmotionalLlama:
    tokenizer: AutoTokenizer
    max_length: int
    emotion_dim: int = EMOTION_DIMENSIONS # Use the constant from emotional_gemma

    def __call__(self, examples: list) -> Dict[str, torch.Tensor]:
        # Separate the components from the examples
        input_ids_list = [example.get("input_ids", []) for example in examples]
        attention_mask_list = [example.get("attention_mask", []) for example in examples]
        emotion_vectors_list = [example.get("emotion_vectors", []) for example in examples]

        # --- Find the token ID for the start of the model's turn ---
        # This is used to mask out user input and padding from the labels
        # Ensure your tokenizer and dataset preparation consistently include this sequence.
        try:
             # Tokenize the specific sequence marking the model's turn start.
             # add_special_tokens=False is crucial here to get just the tokens for the string.
             model_prompt_tokens = self.tokenizer(
                 "<start_of_turn>model\n",
                 add_special_tokens=False
             ).input_ids
             if not model_prompt_tokens:
                 raise ValueError("Tokenizer produced empty list for model prompt sequence.")
             # print(f"DEBUG: Detected model prompt start tokens: {model_prompt_tokens} (decoded: '{self.tokenizer.decode(model_prompt_tokens)}')")
        except Exception as e:
             print(f"ERROR: Could not tokenize model prompt '<start_of_turn>model\\n'. Check tokenizer and template format. Error: {e}")
             raise ValueError("Cannot proceed without identifying model start tokens for label masking.") from e

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_emotion_vectors = []

        # Process each example in the batch
        for i in range(len(input_ids_list)):
            input_ids = input_ids_list[i]
            attention_mask = attention_mask_list[i]
            emotion_vectors = emotion_vectors_list[i]

            # --- Padding ---
            seq_len = len(input_ids)
            pad_len = self.max_length - seq_len

            # Truncate if sequence is longer than max_length (should ideally be handled in dataset)
            if pad_len < 0:
                 input_ids = input_ids[:self.max_length]
                 attention_mask = attention_mask[:self.max_length]
                 emotion_vectors = emotion_vectors[:self.max_length]
                 seq_len = self.max_length
                 pad_len = 0 # Recalculate pad_len after truncation

            # Pad input IDs, attention mask, and emotion vectors
            padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            padded_attention_mask = attention_mask + [0] * pad_len
            # Pad emotion vectors with zero vectors
            padded_emotion_vectors = emotion_vectors + [[0.0] * self.emotion_dim] * pad_len

            # --- Create Labels and Mask User/Padding Tokens ---
            labels = list(padded_input_ids) # Start with a copy of input_ids for labels

            # Find the start index of the model's response to mask previous tokens
            model_start_idx = -1
            # Search for the model prompt token sequence within the original input_ids
            for k in range(seq_len - len(model_prompt_tokens) + 1):
                 if input_ids[k : k + len(model_prompt_tokens)] == model_prompt_tokens:
                      model_start_idx = k
                      break

            if model_start_idx != -1:
                # Mask everything before and including the model's prompt sequence
                for j in range(model_start_idx + len(model_prompt_tokens)):
                    labels[j] = -100
            else:
                print(f"Warning: Model prompt sequence not found in sample {i}. Masking all labels.")
                labels = [-100] * self.max_length # Mask everything

            # Mask padding tokens regardless of model prompt finding
            for j in range(seq_len, self.max_length): # Only mask the padded part
                 labels[j] = -100

            # Sanity check: ensure all lists have the correct length
            if len(padded_input_ids) != self.max_length or \
               len(padded_attention_mask) != self.max_length or \
               len(labels) != self.max_length or \
               len(padded_emotion_vectors) != self.max_length:
                raise ValueError(f"Length mismatch in collator for sample {i} after padding/truncation!")

            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_attention_mask)
            batch_labels.append(labels)
            batch_emotion_vectors.append(padded_emotion_vectors)

        # Convert lists to tensors
        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "emotion_vector": torch.tensor(batch_emotion_vectors, dtype=torch.float),
        }

        return batch


# Subclass Trainer to potentially customize dataloader behavior
class CustomTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        """
        Overrides the method to explicitly use the provided data collator.
        This is mostly for clarity or if the default Trainer behavior needs bypassing.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # Use the data_collator provided during Trainer initialization
        data_collator = self.data_collator

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,  # Important for training
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

def train_emotional_llama(
    model_name=MODEL_NAME, # Use the default model name from emotional_gemma.py
    dataset_path="./dataset.json", # Path to your dataset file
    output_dir="./emotional-gemma-output", # Directory to save results
    max_length=128, # Max sequence length for training
    learning_rate=1e-4, # Base learning rate for LoRA
    emotion_proj_lr=2e-3, # Higher learning rate for emotion projection layer
    num_train_epochs=2,
    per_device_batch_size=12,
    gradient_accumulation_steps=1,
    use_lora=True # Whether to use LoRA
):
    """
    Sets up and runs the training for the EmotionalLlamaModel.
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Set pad_token to eos_token for Gemma if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set padding side to right for causal models
    tokenizer.padding_side = "right"

    print(f"Loading base model: {model_name}")
    # Load the custom EmotionalLlamaModel
    model = EmotionalLlamaModel.from_pretrained(model_name)

    if use_lora:
        print("Applying LoRA configuration")
        # Define LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32, # LoRA rank
            lora_alpha=32, # LoRA scaling factor
            # lora_dropout=0.05, # Dropout for LoRA layers (optional)
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Modules to apply LoRA to
        )
        # Get the PEFT model by wrapping the base model
        model = get_peft_model(model, peft_config)
        # Print trainable parameters summary
        model.print_trainable_parameters()

    # Ensure the emotion projection layer is trainable
    # This is necessary if LoRA was applied, as LoRA defaults other layers to not trainable.
    print("Setting emotion_proj_embed requires_grad=True")
    for param in model.emotion_proj_embed.parameters():
        param.requires_grad = True

    # --- Load and Prepare Dataset ---
    print(f"Loading dataset from: {dataset_path}")
    # Import and use your dataset creation function
    try:
        from dataset import create_huggingface_dataset
        dataset = create_huggingface_dataset(dataset_path, tokenizer, max_length)
        print(f"Dataset loaded with {len(dataset)} examples.")
    except ImportError:
        print("Error: Could not import 'create_huggingface_dataset' from dataset.py")
        print("Please ensure dataset.py exists and contains the necessary function.")
        print("Example dummy dataset creation:")
        # --- PLACEHOLDER! Dummy Dataset Creation Example ---
        # PLACEHOLDER! if dataset.py is not available.
        # Replace this section with your actual dataset loading and processing logic.
        dummy_data = [
            {"text": "<start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\nHi there!", "emotion_vectors": [[0.1]*EMOTION_DIMENSIONS] * 20},
            {"text": "<start_of_turn>user\nHow are you?<end_of_turn>\n<start_of_turn>model\nI'm feeling good today.", "emotion_vectors": [[0.8]*EMOTION_DIMENSIONS] * 25},
        ]
        def dummy_process(example):
            # Simple tokenization for dummy data
            tokenized = tokenizer(example["text"], truncation=True, max_length=max_length, padding="max_length")
            tokenized["emotion_vectors"] = example["emotion_vectors"][:max_length] # Truncate/pad emotion vectors too
            if len(tokenized["emotion_vectors"]) < max_length:
                 tokenized["emotion_vectors"] += [[0.0] * EMOTION_DIMENSIONS] * (max_length - len(tokenized["emotion_vectors"]))
            return tokenized

        from datasets import Dataset
        dataset = Dataset.from_list(dummy_data).map(dummy_process)
        print("Created a dummy dataset. REPLACE THIS with your actual dataset loading!")
        # --- End Dummy Dataset Example ---

    # Initialize the data collator
    data_collator = DataCollatorForEmotionalLlama(tokenizer=tokenizer, max_length=max_length)

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps, # Accumulate gradients over steps
        warmup_ratio=0.1, # Linear warmup over the first 10% of steps
        weight_decay=0.01, # L2 regularization for most parameters
        logging_steps=10, # Log training progress every N steps
        save_steps=200, # Save checkpoint every N steps
        save_total_limit=2, # Keep only the last N checkpoints
        report_to="none", # Disable reporting to external platforms like W&B
        push_to_hub=False, # Do not push to Hugging Face Hub
        bf16=torch.cuda.is_bf16_supported(), # Use bf16 if supported
        fp16=not torch.cuda.is_bf16_supported(), # Otherwise use fp16
        lr_scheduler_type="cosine", # Cosine annealing learning rate scheduler
        optim="adamw_torch" # PyTorch AdamW optimizer
    )

    # --- Optimizer Setup ---
    # Split parameters for different learning rates and weight decay
    # LoRA parameters and other model parameters (if any are trainable beyond LoRA)
    main_params = [p for n, p in model.named_parameters() if p.requires_grad and "emotion_proj" not in n]
    # Emotion projection layer parameters
    emotion_params = [p for n, p in model.named_parameters() if "emotion_proj" in n and p.requires_grad]

    # Define parameter groups for the optimizer
    optimizer_grouped_parameters = [
        # Group for main parameters (LoRA, etc.) with weight decay
        {"params": main_params, "lr": training_args.learning_rate, "weight_decay": training_args.weight_decay},
        # Group for emotion projection layer parameters with a higher LR and NO weight decay
        {"params": emotion_params, "lr": emotion_proj_lr, "weight_decay": 0.0}
    ]

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # --- Initialize Trainer ---
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None), # Pass the custom optimizer
    )

    # --- Optional: Debugging Prints for Dataloader ---
    # print("\n--- Debugging Data Collator Output (First Batch) ---")
    # for step, batch in enumerate(trainer.get_train_dataloader()):
    #     print(f"  Step {step + 1}:")
    #     print(f"    input_ids shape: {batch['input_ids'].shape}")
    #     print(f"    attention_mask shape: {batch['attention_mask'].shape}")
    #     print(f"    emotion_vector shape: {batch['emotion_vector'].shape}")
    #     print(f"    labels shape: {batch['labels'].shape}")
    #     # Print slices or stats for verification
    #     # print(f"    input_ids (first row): {batch['input_ids'][0]}")
    #     # print(f"    labels (first row): {batch['labels'][0]}")
    #     # print(f"    emotion_vector (first row, few elements): {batch['emotion_vector'][0, :10, :2]}")
    #     print(f"    emotion_vector batch MIN: {batch['emotion_vector'].min()}")
    #     print(f"    emotion_vector batch MAX: {batch['emotion_vector'].max()}")
    #     print(f"    emotion_vector batch MEAN: {batch['emotion_vector'].mean()}")
    #     break # Only print the first batch for debug
    # print("--- End Debugging Data Collator Output ---\n")
    # --- End Debugging Prints ---


    # --- Start Training ---
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # --- Save the Model ---
    # Trainer.save_model saves the full model checkpoint by default.
    # If using PEFT, model.save_pretrained() saves only the adapter weights.
    # We want to save BOTH the PEFT adapter and the custom layer weights.

    # Save the PEFT adapter weights if using LoRA
    if use_lora:
        print(f"Saving PEFT adapter model to {output_dir}")
        # This saves adapter_model.safetensors and adapter_config.json
        model.save_pretrained(output_dir)
    else:
        # If not using LoRA, save the full model checkpoint
        print(f"Saving full model checkpoint to {output_dir}")
        trainer.save_model(output_dir)

    # Manually Save Custom Layer Weights (the emotion_proj_embed layer)
    print(f"Saving custom emotion_proj_embed weights...")
    # Access the custom layer, handling the case if the model is wrapped by PEFT
    if hasattr(model, "base_model"): # Check if it's a PeftModel
        emotion_layer = model.base_model.emotion_proj_embed
    else: # If not using PEFT, the layer is directly on the model
        emotion_layer = model.emotion_proj_embed

    # Get the state dictionary of the custom layer
    emotion_state_dict = emotion_layer.state_dict()
    # Define the save path within the output directory
    save_path_emotion = os.path.join(output_dir, "emotion_proj_weights.pth")
    # Save the state dictionary
    torch.save(emotion_state_dict, save_path_emotion)
    print(f"Custom emotion_proj_embed weights saved to: {save_path_emotion}")

    # Return the trained model and tokenizer
    return model, tokenizer

if __name__ == "__main__":
    # Make sure you have a dataset.py and dataset.json file or implement the dummy dataset creation above.
    # Replace the dataset_path with the actual path to your dataset.
    train_emotional_llama(
        dataset_path="./dataset.json", # Replace with your dataset path
        output_dir="./emotional-gemma-output", # Output directory
        max_length=128,
        num_train_epochs=3,
        per_device_batch_size=4, # Adjust based on your GPU memory
        gradient_accumulation_steps=8, # Adjust based on desired effective batch size
        learning_rate=2e-4, # Base LR for LoRA
        emotion_proj_lr=5e-3, # Higher LR for emotion layer
        use_lora=True
    )