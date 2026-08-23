import os
from transformers import utils
from huggingface_hub import constants

# Force the cache directory for both libraries
os.environ["HF_HOME"] = "/home/users/kagyemang-dua/huggingface_cache"
utils.TRANSFORMERS_CACHE = "/home/users/kagyemang-dua/huggingface_cache"
constants.HF_HOME = "/home/users/kagyemang-dua/huggingface_cache"
constants.HF_HUB_CACHE = "/home/users/kagyemang-dua/huggingface_cache/hub"
