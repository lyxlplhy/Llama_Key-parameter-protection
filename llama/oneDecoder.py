from transformers import LlamaPreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaSdpaAttention,LlamaRMSNorm,LlamaAttention,LlamaFlashAttention2
import torch.nn as nn
import torch
from typing import  Optional, Tuple

LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class Att_inputlayernorm(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        outputs = (hidden_states,residual,self_attn_weights,present_key_value)
        return outputs


class oneDecoder(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers0_atten=Att_inputlayernorm(config, layer_idx=0)
        
    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        position_ids = torch.arange(0, x.shape[1], dtype=torch.long).unsqueeze(0).expand(x.size(0), -1).to(x.device)
        x=self.layers0_atten(x,position_ids=position_ids)
        return x

def main():
    #load model    
    base_path = "/gemini/data-1/model_base/Llama-2-7b-chat-hf"
    test_model = oneDecoder(LlamaConfig.from_pretrained(base_path))
    embedding_state_dict = torch.load('/gemini/data-3/model_base/llama-2-7b-1/embedding_state_dict.pt')
    test_model.embed_tokens.load_state_dict(embedding_state_dict)
    test_model.layers0_atten.input_layernorm.load_state_dict(torch.load('/gemini/data-3/model_base/llama-2-7b-1/layers0_input_layernorm.pt'))
    test_model.layers0_atten.self_attn.load_state_dict(torch.load('/gemini/data-3/model_base/llama-2-7b-1/layers0_self_attn.pt'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_model.to(device)

    #infernece
    messages = [
    {"role": "user", "content": "Translate the following content into Chinese: Hello"},
]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True,device_map='auto')
    input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(test_model.device)



    output_ids = test_model(input_ids)  
    print(output_ids)

    #save_result
    torch.save(output_ids, '/gemini/code/mistral_finetune_code/tee/tee2/on_out/output_ids.pt')

if __name__ == "__main__":
    main()
