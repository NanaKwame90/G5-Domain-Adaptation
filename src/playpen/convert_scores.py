import pandas as pd
import json

# Load the saved scores
with open('results/Qwen3.5-9B/scores.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV and HTML
df.to_csv('Qwen3.5-9B_scores.csv', index=False)
df.to_html('Qwen3.5-9B_scores.html', index=False)

print("Converted successfully.")
