import json
import os

def split_instances(input_file, output_dir):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for idx, instance in enumerate(data):
        # Format filename to match the expected pattern
        filename = f"instance_{idx:05d}.json"
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            json.dump(instance, f, indent=4)
    print(f"Split {len(data)} instances into {output_dir}")

# Adjust paths as needed
split_instances('/home/users/kagyemang-dua/clembench_repo/adventuregame/in/instances.json', 
                '/home/users/kagyemang-dua/clembench_repo/adventuregame/in/')
