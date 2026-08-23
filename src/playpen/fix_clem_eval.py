import os

clemeval_path = "/home/users/kagyemang-dua/g5_adaptation/venv/lib/python3.10/site-packages/clemcore/clemeval.py"

if os.path.exists(clemeval_path):
    with open(clemeval_path, "r") as f:
        content = f.read()

    # The strict unpacking line causing the error:
    # model, game, experiment, episode, _ = splits[-5:]
    
    old_code = "model, game, experiment, episode, _ = splits[-5:]"
    new_code = """# Patched to handle varying split lengths safely
    if len(splits) >= 5:
        model, game, experiment, episode, _ = splits[-5:]
    else:
        # Fallback padding if path segments are fewer than 5
        padded = ['unknown'] * (5 - len(splits)) + list(splits)
        model, game, experiment, episode, _ = padded[-5:]"""

    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(clemeval_path, "w") as f:
            f.write(content)
        print("Successfully patched clemcore/clemeval.py!")
    else:
        print("Could not find the target line to patch. It might already be modified.")
else:
    print("clemeval.py path not found.")
