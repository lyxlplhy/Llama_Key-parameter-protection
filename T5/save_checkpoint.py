from transformers import AutoTokenizer,T5ForConditionalGeneration
import torch

tokenizer = AutoTokenizer.from_pretrained(
    '/gemini/code/checkpoint/t5-3b'
    )
model = T5ForConditionalGeneration.from_pretrained(
    '/gemini/code/checkpoint/t5-3b',
    # torch_dtype=torch.float16,
    device_map="auto"
    )
print(model)


torch.save(model.shared.state_dict(), '/gemini/data-3/T5/shared.pt')
torch.save(model.encoder.final_layer_norm.state_dict(),"/gemini/data-3/T5/encoder_final_layer_norm.pt")
torch.save(model.encoder.embed_tokens.state_dict(), '/gemini/data-3/T5/encoder_embed_tokens.pt')
for i in range(len(model.encoder.block)):
    torch.save(model.encoder.block[i].state_dict(), f'/gemini/data-3/T5/encoder_layers{i}.pt')

torch.save(model.decoder.state_dict(), '/gemini/data-3/T5/decoder_all.pt')

torch.save(model.lm_head.state_dict(), '/gemini/data-3/T5/lm_head.pt')