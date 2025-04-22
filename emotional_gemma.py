# emotional_gemma.py 
import torch
import torch.nn as nn
from transformers import Gemma3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Union

# Constants
MODEL_NAME = "google/gemma-3-1b-it"
EMOTION_DIMENSIONS = 8
EMOTION_DIMENSIONS_REFERENCE = [
    "SADNESS_JOY", "FEAR_COURAGE", "DISGUST_ACCEPTANCE", "ANGER_CALMNESS",
    "SURPRISE_EXPECTATION", "DISTRUST_TRUST", "BOREDOM_INTEREST", "INDIFFERENCE_EMPATHY"
]

class EmotionalLlamaModel(Gemma3ForCausalLM):
    """Gemma3 Causal Language Model with emotion modulation."""
    def __init__(self, config):
        super().__init__(config)
        self.emotion_dim = EMOTION_DIMENSIONS

        # Emotion projection layer: MLP
        # This layer projects the emotion vector to the hidden size of the model.
        intermediate_size = config.hidden_size // 2
        self.emotion_proj_embed = nn.Sequential(
            nn.Linear(self.emotion_dim, intermediate_size),
            nn.LayerNorm(intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, config.hidden_size),
        )

        # Initialization for the MLP weights
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        self.emotion_proj_embed.apply(init_weights)

        # Post-initialization steps from the base class
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        emotion_vector: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:

        # 1. Prepare Input Embeddings
        # Get input embeddings from input_ids or use provided inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_len = input_ids.shape
            inputs_embeds = self.model.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
        else:
             # If neither is provided, it's likely a generation step using only cache.
             # The base model's forward handles this by looking up the single new token ID.
             # We will rely on the base model forward to handle this case and potentially
             # receive `inputs_embeds` as `kwargs`.
             pass # Standard generate handle embedding lookup for subsequent tokens


        # 2. Apply Emotion Modulation to Embeddings
        # If emotion_vector is provided and we have inputs_embeds, modulate the embeddings
        if emotion_vector is not None and inputs_embeds is not None:
            if emotion_vector.shape[0] != batch_size:
                raise ValueError("Batch size mismatch between emotion_vector and input.")

            # Ensure emotion_vector shape is [batch, seq_len, emotion_dim]
            # This handles the case where a single emotion vector [batch, emotion_dim]
            # is provided for the entire sequence during inference.
            current_seq_len = inputs_embeds.shape[1]
            if emotion_vector.dim() == 2:
                emotion_vector = emotion_vector.unsqueeze(1).expand(-1, current_seq_len, -1)
            elif emotion_vector.shape[1] != current_seq_len:
                # This case might occur if the emotion vector is longer than the current
                # input chunk (e.g., during token-by-token generation after prompt).
                # We take the slice corresponding to the current input.
                 emotion_vector = emotion_vector[:, :current_seq_len, :]


            # Project emotion vector to hidden size using the emotion projection layer
            emotion_offset = self.emotion_proj_embed(emotion_vector) # -> [batch, current_seq_len, hidden_size]

            # Add the projected emotion vector as an offset to the input embeddings
            # Scaling factor (e.g., 3) can be adjusted during training
            inputs_embeds = inputs_embeds + emotion_offset * 3

        # 3. Pass embeddings (potentially modified) to the base model's core layers
        # Crucially, pass inputs_embeds if they were modified, otherwise input_ids
        # (though the base forward handles input_ids -> inputs_embeds)
        outputs = self.model(
            input_ids=input_ids if inputs_embeds is None else None, # Pass input_ids ONLY if inputs_embeds wasn't created/modified
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, # Always pass the potentially modified inputs_embeds
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True, # Need last hidden state for lm_head
            return_dict=True,
            **kwargs
        )

        # 4. Compute logits from the final hidden state
        hidden_states = outputs.hidden_states[-1]

        # Apply the language model head to get logits
        logits = self.lm_head(hidden_states)

        # 5. Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift tokens for autoregressive training
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        # Return the CausalLMOutputWithPast object
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, # Optionally keep all hidden states
            attentions=outputs.attentions, # Optionally keep attentions
        )

# This file only contains the model definition and constants.
# Training and inference logic are handled in separate files.