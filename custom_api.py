import os
os.environ['HF_HOME'] = '/home/users/kagyemang-dua/.cache/huggingface'
os.environ['HF_HUB_CACHE'] = '/home/users/kagyemang-dua/.cache/huggingface/hub'
os.environ['TRANSFORMERS_CACHE'] = '/home/users/kagyemang-dua/.cache/huggingface/transformers'

from huggingface_hub import constants
constants.HF_HUB_CACHE = os.environ['HF_HUB_CACHE']

import torch
from peft import PeftModel
import clemcore.backends.huggingface_local_api as hf_backend


class Backend(hf_backend.HuggingfaceLocal):
    """Custom backend that loads a base HuggingfaceLocal model and applies
    a PEFT/LoRA adapter on top of it."""

    ADAPTER_PATH = "/home/users/kagyemang-dua/g5_adaptation/kto_qwen_stable_model"

    def get_model_for(self, model_spec):
        # Let the parent build the base HuggingfaceLocalModel
        # (this is where self.tokenizer, self.config, self.model get set)
        model = super().get_model_for(model_spec)

        # Apply the LoRA adapter on top of the already-loaded base model
        model.model = PeftModel.from_pretrained(model.model, self.ADAPTER_PATH)

        if torch.cuda.is_available():
            model.model = model.model.to("cuda")

        model.model.eval()
        return model
