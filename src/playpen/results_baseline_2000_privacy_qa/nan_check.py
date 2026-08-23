from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Qwen/Qwen3.5-9B"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

for dtype_name, dtype in [("bfloat16", torch.bfloat16), ("float32", torch.float32)]:
    print(f"\n=== Testing dtype: {dtype_name} ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()

    messages = [{"role": "user", "content": "Sort these words alphabetically: syndrome therefrom"}]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[0, -1, :]  # logits for next token after prompt

    print("logits dtype:", logits.dtype)
    print("any NaN:", torch.isnan(logits).any().item())
    print("any Inf:", torch.isinf(logits).any().item())
    print("logits min/max:", logits.min().item(), logits.max().item())

    top5 = torch.topk(logits.float(), 5)
    print("top5 token ids:", top5.indices.tolist())
    print("top5 tokens:", [tok.decode([t]) for t in top5.indices.tolist()])
    print("top5 values:", top5.values.tolist())

    del model
    torch.cuda.empty_cache()
