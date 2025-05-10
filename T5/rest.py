from T5_model import T5Stack_onelayer
from transformers.models.t5.modeling_t5 import T5PreTrainedModel,T5Config,T5Block,T5LayerNorm
import torch
import torch.nn as nn
import copy
from T5_model import *

def main():
    #load_model
    base_path = "/gemini/code/checkpoint/t5-3b"
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True,device_map='auto')
    test_model = T5_rest(config=T5ForConditionalGeneration.from_pretrained('/gemini/code/checkpoint/t5-3b').config)
    test_model.shared.load_state_dict(torch.load('/gemini/data-3/T5/shared.pt'))
    test_model.encoder.embed_tokens.load_state_dict(torch.load("/gemini/data-3/T5/encoder_embed_tokens.pt"))
    test_model.encoder.final_layer_norm.load_state_dict(torch.load("/gemini/data-3/T5/encoder_final_layer_norm.pt"))
    for i in range(24):
        test_model.encoder.block[i].load_state_dict(torch.load(f"/gemini/data-3/T5/encoder_layers{i}.pt"))
    del test_model.encoder.block[0]
    test_model.decoder.load_state_dict(torch.load(f"/gemini/data-3/T5/decoder_all.pt"))
    test_model.lm_head.load_state_dict(torch.load("/gemini/data-3/T5/lm_head.pt"))
    test_model = test_model.to(torch.float16)
    test_model.eval()
    test_model.to(torch.device("cuda:0"))

    #load input 
    loaded_output_ids = torch.load('/gemini/code/mistral_finetune_code/tee/T5/output_ids.pt')#one的输出
    loaded_output_ids.to(test_model.device)
    loaded_output_ids2 = torch.load('/gemini/code/mistral_finetune_code/tee/T5/position_bias.pt')#one的输出
    loaded_output_ids2.to(test_model.device)
    inputs = torch.load('/gemini/code/mistral_finetune_code/tee/inputs.pt')
    inputs.to(test_model.device)
    decoder_input_ids = torch.tensor([[test_model.config.decoder_start_token_id]], device=test_model.device)
    max_new_tokens = 50
    encoder_outputs=None

    #inference
    for _ in range(max_new_tokens):

        outputs = test_model(
            input_ids=inputs["input_ids"],
            hidden_states_temp=loaded_output_ids,
            position_bias=loaded_output_ids2,
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
        use_cache=False  # 关闭 cache 简化逻辑
    )
        encoder_outputs=outputs.encoder_hidden_states
        next_token_logits = outputs.logits[:, -1, :] 
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)  
        if next_token_id.item() == tokenizer.eos_token_id:
            break
        decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)
    generated_text = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
    print(generated_text)

if __name__ == "__main__":
    main()
