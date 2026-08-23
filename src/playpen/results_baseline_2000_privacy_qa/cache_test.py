from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Qwen/Qwen3.5-9B"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model.eval()

messages = [{"role": "user", "content": "Sort these words alphabetically: syndrome therefrom"}]
prompt = tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inputs = tok(prompt, return_tensors="pt").to(model.device)

print("=== use_cache=True (default) ===")
out = model.generate(**inputs, max_new_tokens=20, do_sample=False, use_cache=True)
print(repr(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)))

print("=== use_cache=False (force full recompute each step) ===")
out2 = model.generate(**inputs, max_new_tokens=20, do_sample=False, use_cache=False)
print(repr(tok.decode(out2[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)))
