import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,T5ForConditionalGeneration
import torch.nn.functional as F
tokenizer = AutoTokenizer.from_pretrained(
    '/gemini/code/checkpoint/t5-3b'
    )
model = T5ForConditionalGeneration.from_pretrained(
    '/gemini/code/checkpoint/t5-3b',
    torch_dtype=torch.float16,
    device_map="auto"
    )

###############################################################
prompt = "translate English to French: The weather is nice today."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# encoder_outputs = model.get_encoder()(**inputs)

decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], device=model.device)


max_new_tokens = 50


for _ in range(max_new_tokens):
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        decoder_input_ids=decoder_input_ids,
        # encoder_outputs=encoder_outputs,
        use_cache=False  # 关闭 cache 简化逻辑
    )

    next_token_logits = outputs.logits[:, -1, :]  # 取最新的一个 token 的 logits
    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # 贪婪选最大概率

    
    if next_token_id.item() == tokenizer.eos_token_id:
        break


    decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)


generated_text = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
print(generated_text)