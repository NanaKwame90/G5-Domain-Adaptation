import json
import os

for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.json'):
            path = os.path.join(root, file)
            try:
                with open(path, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                print(f"BROKEN FILE FOUND: {path}")
                print(f"Error details: {e}")
            except Exception as e:
                print(f"OTHER ERROR: {path} - {e}")
