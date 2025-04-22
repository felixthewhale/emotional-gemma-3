# inference.py
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
from transformers import AutoTokenizer
from emotional_gemma import EmotionalLlamaModel, EMOTION_DIMENSIONS, EMOTION_DIMENSIONS_REFERENCE
from peft import PeftModel, PeftConfig

import torch.nn.functional as F


def generate_with_emotion(
    model,
    tokenizer,
    prompt: str,
    emotion_vector: list,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 128,
    top_p: float = 0.95,
    do_sample: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = None,
):
    """
    Generates text using the standard model.generate() method with an emotion vector.
    """
    print(f"Generation parameters: max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k}, top_p={top_p}, do_sample={do_sample}")
    if len(emotion_vector) != EMOTION_DIMENSIONS:
        raise ValueError(f"Emotion vector must have {EMOTION_DIMENSIONS} dimensions.")

    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

    current_model = model
    current_model.eval()
    current_model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    # Emotion vector needs to be a tensor and moved to the correct device
    emotion_tensor = torch.tensor([emotion_vector], dtype=torch.float).to(device) # Shape [1, EMOTION_DIMENSIONS]

    with torch.no_grad():
        # Pass the emotion vector to the generate method
        generated_outputs = current_model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            emotion_vector=emotion_tensor, # Pass the [1, EMOTION_DIMENSIONS] tensor
        )

    generated_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
    return generated_text

# --- Main block ---
if __name__ == "__main__":
    # Directory where the adapter weights and custom layer weights were saved
    model_path = "./emotional-gemma-output-4"

    # --- Load configuration ---
    # PEFT config should tell us the base model name
    try:
        config = PeftConfig.from_pretrained(model_path)
        model_name = config.base_model_name_or_path
        print(f"Inferred base model name from PEFT config: {model_name}")
    except Exception as e:
        print(f"Warning: Could not infer base model name from PeftConfig in {model_path}. Using default. Error: {e}")
        # Fallback if config loading fails
        model_name = "google/gemma-3-1b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Load the base model ---
    # The base model needs to be the custom EmotionalLlamaModel
    print(f"Loading base model: {model_name}")
    base_model = EmotionalLlamaModel.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    print("Base model loaded.")

    # --- Load the PEFT model (adapter weights only) ---
    print(f"Loading PEFT adapter from: {model_path}")
    # This wraps the base_model with PEFT adapters
    model = PeftModel.from_pretrained(base_model, model_path)
    print(f"PEFT adapter loaded. Model type: {type(model)}")

    # --- Explicitly Load Custom Layer Weights ---
    # Load the state_dict for the custom layer from the saved file
    custom_weights_path = os.path.join(model_path, "emotion_proj_weights.pth")
    try:
        if os.path.exists(custom_weights_path):
            print(f"Loading custom emotion_proj_embed weights from: {custom_weights_path}")
            # Load the state dict, mapping to CPU first is safer before loading into model
            emotion_state_dict = torch.load(custom_weights_path, map_location="cpu")

            # Access the layer within the PeftModel's base_model
            # The custom layer is directly on the base model instance
            emotion_layer = model.base_model.emotion_proj_embed
            load_result = emotion_layer.load_state_dict(emotion_state_dict)
            print(f"Custom weights loaded successfully: {load_result}")
        else:
            print(f"WARNING: Custom weights file not found at {custom_weights_path}. Layer 'emotion_proj_embed' will have base model's initial weights.")

    except Exception as e:
        print(f"ERROR loading custom emotion_proj_embed weights from {custom_weights_path}: {e}")

    # Determine and move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Moving model to device: {device}")
    model.to(device)


    # --- Model Weight Checks (After Loading) ---
    print("\n--- Model Weight Checks (After Loading) ---")
    is_peft_model = isinstance(model, PeftModel)
    print(f"Is PeftModel: {is_peft_model}")

    print("  emotion_proj Layer Check:")
    try:
        # Access the custom layer via the base_model attribute of the PeftModel
        emotion_proj_layer = model.base_model.emotion_proj_embed
        print(f"    - emotion_proj_embed Sequential found: {emotion_proj_layer}")
        # Assuming the Sequential contains a Linear layer at index 0
        linear_layer = emotion_proj_layer[0]
        print(f"    - Linear layer inside Sequential: {linear_layer}")
        if hasattr(linear_layer, 'weight'):
             print(f"      Weights exist, device: {linear_layer.weight.device}, dtype: {linear_layer.weight.dtype}")
             print(f"      Weights mean abs value: {linear_layer.weight.data.abs().mean().item()}")
        else: print("      Weights attribute not found.")
        if hasattr(linear_layer, 'bias') and linear_layer.bias is not None:
             print(f"      Bias exist, device: {linear_layer.bias.device}, dtype: {linear_layer.bias.dtype}")
             print(f"      Bias mean abs value: {linear_layer.bias.data.abs().mean().item()}")
        else: print("      Bias attribute not found or is None.")
    except Exception as e: print(f"    - Error checking layer: {e}")

    # Check the device of one of the model parameters
    print(f"Model overall device: {next(model.parameters()).device}")

    # --- Generation ---
    # Prepare the prompt using the chat template
    prompt = tokenizer.apply_chat_template([
        {"role": "user", "content": "You are a program, not a person"},
    ], tokenize=False, add_generation_prompt=True)

    print(f"\nPrompt:\n{prompt}")

    # Define emotion vectors based on the reference dimensions
    # EMOTION_DIMENSIONS_REFERENCE is defined in emotional_gemma.py
    # Index mapping: 0=SADNESS_JOY, 1=FEAR_COURAGE, 2=DISGUST_ACCEPTANCE, 3=ANGER_CALMNESS,
    # 4=SURPRISE_EXPECTATION, 5=DISTRUST_TRUST, 6=BOREDOM_INTEREST, 7=INDIFFERENCE_EMPATHY
    joyful_emotion = [0.8, 0, 0, 0, 0, 0, 0, 0] # High Joy, some Calmness
    sad_emotion = [-0.8, 0, 0, 0, 0, 0, 0, 0]  # High Sadness, some Calmness
    neutral_emotion = [0] * EMOTION_DIMENSIONS # All dimensions at zero
    my_seed = 42 # Seed for reproducibility

    # Generate text with different emotions using the recommended method
    print("Generating with joyful emotion:")
    joyful_text = generate_with_emotion(model, tokenizer, prompt, joyful_emotion, seed=my_seed)
    print(joyful_text)

    print("\nGenerating with sad emotion:")
    sad_text = generate_with_emotion(model, tokenizer, prompt, sad_emotion, seed=my_seed)
    print(sad_text)

    print("\nGenerating with neutral emotion:")
    neutral_text = generate_with_emotion(model, tokenizer, prompt, neutral_emotion, seed=my_seed)
    print(neutral_text)