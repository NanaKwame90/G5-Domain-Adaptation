import json

# This expects raw_logs.jsonl to have 'prompt', 'response', and 'is_correct' keys
with open("raw_logs.jsonl", "r") as f_in, open("train_kto.jsonl", "w") as f_out:
    for line in f_in:
        entry = json.loads(line)
        output = {
            "prompt": entry["prompt"],
            "completion": entry["response"],
            "label": entry["is_correct"]
        }
        f_out.write(json.dumps(output) + "\n")
