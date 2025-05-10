import json
import torch
import numpy as np
from transformers import AutoModel,AutoTokenizer,AutoModelForCausalLM,LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.llama.modeling_llama import LlamaModel
from peft import LoraConfig, get_peft_model,PeftModel,PeftConfig
# from transformers.cache_utils import  DynamicCache
import torch.nn.functional as F
import torch.nn as nn
import sys
import os
base_path = "/gemini/data-1/model_base/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True,device_map='auto')
base_model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True,device_map='auto')
print(base_model)

torch.save(base_model.model.embed_tokens.state_dict(), '/gemini/data-3/model_base/llama-2-7b-1/embedding_state_dict.pt')

for i in range(len(base_model.model.layers)):
    if i==0:
        torch.save(base_model.model.layers[i].state_dict(), f'/gemini/data-3/model_base/llama-2-7b-1/layers{i}.pt')
        torch.save(base_model.model.layers[0].self_attn.state_dict(), f'/gemini/data-3/model_base/llama-2-7b-1/layers0_self_attn.pt')
        torch.save(base_model.model.layers[0].mlp.state_dict(), f'/gemini/data-3/model_base/llama-2-7b-1/layers0_mlp.pt')
        torch.save(base_model.model.layers[0].input_layernorm.state_dict(), f'/gemini/data-3/model_base/llama-2-7b-1/layers0_input_layernorm.pt')
        torch.save(base_model.model.layers[0].post_attention_layernorm.state_dict(), f'/gemini/data-3/model_base/llama-2-7b-1/layers0_post_attention_layernorm.pt')
 
    else:
        torch.save(base_model.model.layers[i].state_dict(), f'/gemini/data-3/model_base/llama-2-7b-1/layers{i}.pt')
torch.save(base_model.model.norm.state_dict(), '/gemini/data-3/model_base/llama-2-7b-1/norm.pt')   

torch.save(base_model.lm_head.state_dict(), '/gemini/data-3/model_base/llama-2-7b-1/lm_head.pt')  



