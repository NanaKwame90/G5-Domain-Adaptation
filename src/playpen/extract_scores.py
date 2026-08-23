import pandas as pd
import json
import os

results_root = 'results/Qwen3.5-9B'
target_files = ['scores.json', 'metrics.json', 'summary.json']
all_data = []

print(f"Searching for {target_files} in {results_root}...")

for root, dirs, files in os.walk(results_root):
    for file in files:
        if file in target_files:
            file_path = os.path.join(root, file)
            print(f"Found: {file_path}")
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        df_part = pd.json_normalize(data)
                    else:
                        df_part = pd.json_normalize([data])
                    df_part['source_file'] = file
                    all_data.append(df_part)
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")

if all_data:
    df = pd.concat(all_data, ignore_index=True)
    df.to_csv('final_scores.csv', index=False)
    df.to_html('final_scores.html', index=False)
    print(f"\nSUCCESS: Exported {len(all_data)} records to final_scores.csv and final_scores.html")
else:
    print("\nNo matching score files found. Check your file names.")
