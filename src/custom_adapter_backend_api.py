from clemcore.backends.huggingface import HuggingFaceLocalBackend
from peft import PeftModel
from transformers import AutoModelForCausalLM

class CustomAdapterBackend(HuggingFaceLocalBackend):
    def __init__(self, model_spec):
        # Initialize the base class
        super().__init__(model_spec)
        
        # Define the path to your adapters
        adapter_path = "/home/users/kagyemang-dua/g5_adaptation/kto_qwen_stable_model"
        
        # Load the adapter on top of the base model
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        
        # Ensure the model is in eval mode
        self.model.eval()

# Register this backend in your clembench environment
# You may need to add this entry to the __init__.py of your backends/ folder
