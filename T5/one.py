from T5_model import T5Stack_onelayer
from transformers.models.t5.modeling_t5 import T5PreTrainedModel,T5Config,T5Block,T5LayerNorm
import torch
import torch.nn as nn
import copy
from T5_model import *
class One_encoder(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.encoder_one = T5Stack_onelayer(encoder_config, self.shared)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs,position_bias = self.encoder_one(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        return encoder_outputs,position_bias


def main():
    #load model
    base_path = "/gemini/code/checkpoint/t5-3b"
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True,device_map='auto')
    test_model = One_encoder(config=T5ForConditionalGeneration.from_pretrained('/gemini/code/checkpoint/t5-3b').config)
    test_model.shared.load_state_dict(torch.load('/gemini/data-3/T5/shared.pt'))
    test_model.encoder_one.embed_tokens.load_state_dict(torch.load("/gemini/data-3/T5/encoder_embed_tokens.pt"))
    test_model.encoder_one.block[0].load_state_dict(torch.load('/gemini/data-3/T5/encoder_layers0.pt'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_model.to(device)
    test_model.eval()
    test_model = test_model.to(torch.float16)

    #infernece
    prompt = "translate English to French: The weather is very nice today."
    inputs = tokenizer(prompt, return_tensors="pt").to(test_model.device)

    torch.save(inputs, 'inputs.pt')
    
    outputs ,position_bias= test_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    torch.save(outputs, 'output_ids.pt')
    torch.save(position_bias, 'position_bias.pt')
if __name__ == "__main__":
    main()
