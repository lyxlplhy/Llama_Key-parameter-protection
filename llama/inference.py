from restModel import restModel
from oneDecoder import oneDecoder
from transformers.models.llama.configuration_llama import LlamaConfig
import torch
from transformers import AutoTokenizer

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_path = "/gemini/data-1/model_base/Llama-2-7b-chat-hf"
    
    #load one_model
    one_model = oneDecoder(LlamaConfig.from_pretrained(base_path))
    embedding_state_dict = torch.load('/gemini/data-3/model_base/llama-2-7b-1/embedding_state_dict.pt')
    one_model.embed_tokens.load_state_dict(embedding_state_dict)
    one_model.layers0_atten.input_layernorm.load_state_dict(torch.load('/gemini/data-3/model_base/llama-2-7b-1/layers0_input_layernorm.pt'))
    one_model.layers0_atten.self_attn.load_state_dict(torch.load('/gemini/data-3/model_base/llama-2-7b-1/layers0_self_attn.pt'))
    one_model.to(device)

    #load rest_model
    rest_model = restModel(LlamaConfig.from_pretrained(base_path))
    rest_model.layers0_mlp.mlp.load_state_dict(torch.load('/gemini/data-3/model_base/llama-2-7b-1/layers0_mlp.pt'))
    rest_model.layers0_mlp.post_attention_layernorm.load_state_dict(torch.load('/gemini/data-3/model_base/llama-2-7b-1/layers0_post_attention_layernorm.pt'))
    for i in range(1, 32):
        rest_model.layers[i - 1].load_state_dict(torch.load(f'/gemini/data-3/model_base/llama-2-7b-1/layers{i}.pt'))
    rest_model.lm_head.load_state_dict(torch.load('/gemini/data-3/model_base/llama-2-7b-1/lm_head.pt'))
    rest_model.norm.load_state_dict(torch.load('/gemini/data-3/model_base/llama-2-7b-1/norm.pt'))
    rest_model.to(device)

    messages = [{"role": "user", "content": "Translate the following content into English: 你好，我今天中午吃了牛肉面"},]   
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True,device_map='auto')
    input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
    ).to(device)
    generated = input_ids.clone()
    max_new_tokens = 256
    temperature = 0.6
    top_p = 0.9
    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    for _ in range(max_new_tokens):
        with torch.no_grad():
            one_output = one_model(input_ids=generated)
            outputs=rest_model(one_output)
            # next_token_logits = outputs.logit[:, -1, :]  # 取最后一个 token 的 logits
            next_token_logits = outputs[:, -1, :]  # ✅ 修正位置
            # Top-p nucleus 采样
            probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # 保留 cumulative prob < top_p 的部分
            cutoff = cumulative_probs > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_probs[cutoff] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(sorted_probs, num_samples=1)
            next_token_id = sorted_indices.gather(-1, next_token)

        # 拼接新的 token
        generated = torch.cat([generated, next_token_id], dim=-1)

        # 判断是否终止
        if next_token_id.item() in terminators:
            break
    generated_text = tokenizer.decode(generated[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print(generated_text)

if __name__ == "__main__":
    main()
