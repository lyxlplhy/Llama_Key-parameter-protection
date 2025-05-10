from transformers import LlamaPreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer,LlamaMLP,LlamaRMSNorm
import torch.nn as nn
import torch
from typing import  Optional, Tuple

class MLP_postattentionlayernorm(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.mlp = LlamaMLP(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],#hidden_states,residual,self_attn_weights,present_key_value
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        hidden_states, residual, self_attn_weights, present_key_value = inputs
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class restModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.layers0_mlp=MLP_postattentionlayernorm(config,layer_idx =0)
        self.layers = nn.ModuleList(
                [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(1,32)]
            )
        num_gpus = torch.cuda.device_count()
        layers_per_gpu = len(self.layers) / num_gpus
        for i in range(len(self.layers)):
            gpu_idx = int(i // layers_per_gpu)
            self.layers[i] = self.layers[i].to(f'cuda:{gpu_idx}')
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(f'cuda:{num_gpus-1}')
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(f'cuda:{num_gpus-1}')
    
    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],#hidden_states,residual,self_attn_weights,present_key_value
        position_ids: Optional[torch.LongTensor] = None,
       
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        x=self.layers0_mlp(inputs,position_ids=0)[0]
        i=1
        for layer in self.layers:
            x = x.to(layer.self_attn.k_proj.weight.device)
            position_ids = torch.arange(0, x.shape[1], dtype=torch.long).unsqueeze(0).expand(x.size(0), -1).to(x.device)
            x = layer(x,position_ids=position_ids)[0]
            print(f'{i}layers',x)
            i+=1
        print(x.shape)
        x = self.norm(x)
        logits = self.lm_head(x)
        probs = torch.softmax(logits, dim=-1)
        next_token_index = torch.argmax(probs, dim=-1)
        return next_token_index

#load model   
base_path = "/gemini/data-1/model_base/Llama-2-7b-hf"
test_model = restModel(LlamaConfig.from_pretrained(base_path))
test_model.layers0_mlp.mlp.load_state_dict(torch.load('/gemini/data-3/model_base/llama-2-7b-1/layers0_mlp.pt'))
test_model.layers0_mlp.post_attention_layernorm.load_state_dict(torch.load('/gemini/data-3/model_base/llama-2-7b-1/layers0_post_attention_layernorm.pt'))
for i in range(1,32):
    test_model.layers[i-1].load_state_dict(torch.load(f'/gemini/data-3/model_base/llama-2-7b-1/layers{i}.pt'))
test_model.lm_head.load_state_dict(torch.load('/gemini/data-3/model_base/llama-2-7b-1/lm_head.pt'))
test_model.norm.load_state_dict(torch.load('/gemini/data-3/model_base/llama-2-7b-1/norm.pt'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_model.to(device)

#load input and infernence
loaded_output_ids = torch.load('/gemini/code/mistral_finetune_code/tee/tee2/on_out/output_ids.pt')
output=test_model(loaded_output_ids)
print(output)

