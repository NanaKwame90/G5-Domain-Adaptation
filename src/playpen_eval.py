# src/playpen_eval.py
import sys
import os

# Add the project root (g5/ directory) to sys.path
# This makes 'src' importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Updated import based on your directory structure: src/playpen/...
# Adjust 'examples.baseline_runner' if the path inside playpen/ differs
from src.playpen.examples import baseline_runner 

def run_baseline():
    print("Initializing Playpen Sandbox...")
    # The framework uses the observe-act-step paradigm to collect game interactions[cite: 3]
    # Ensure 'Qwen3.5-9B' matches the model_name you defined in model_registry.json
    runner = baseline_runner.setup_eval(model_name="Qwen3.5-9B")
    
    print("Starting baseline evaluation...")
    # These game strings must match the game identifiers in the clembench framework
    results = runner.play_games(games=["taboo", "wordle", "referencegame"])
    
    print("Baseline Complete. Saving results...")
    # Placeholder for your logging logic
    return results

if __name__ == "__main__":
    run_baseline()