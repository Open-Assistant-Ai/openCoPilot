# write a python3 script that will launch the following project https://github.com/huggingface/text-generation-inference

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import os

# Define the model name and the output directory
MODEL_NAME = "flax-community/gpt-neo-125M-code-clippy-dedup-2048"
OUTPUT_DIR = "codeClippy"

# Create the output directory if it does not exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load the tokenizer and the model
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPTNeoForCausalLM.from_pretrained(MODEL_NAME)

# Save the tokenizer and the model to the output directory
tokenizer.save_pretrained(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
