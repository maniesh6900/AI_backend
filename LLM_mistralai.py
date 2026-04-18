"""
HuggingFace Model Connection Script
Supports: openai-community/gpt2, and other OSS models from HuggingFace Hub

Fixes applied:
  - SSL_CERT_FILE crash on Windows
  - torch_dtype -> dtype (deprecation fix)
  - low_cpu_mem_usage=True  (prevents segfault on large models / low RAM)
  - Optional 4-bit quantisation for very large models (requires bitsandbytes)
"""

import os

# ── Fix: Windows SSL_CERT_FILE pointing to a missing path ──
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
import torch
from dotenv import load_dotenv
load_dotenv()

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MODEL_ID = os.getenv("MODEL")   # Change to any HF model ID
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("TOKEN") # Needed for gated/private models

# Set True to load large models in 4-bit (saves VRAM/RAM, requires bitsandbytes)
# pip install bitsandbytes
USE_4BIT = False


def load_model(model_id: str = MODEL_ID):
    """Load tokenizer and model from HuggingFace Hub."""
    print(f"Loading model : {model_id}")
    print(f"Device        : {DEVICE}")
    print(f"4-bit quant   : {USE_4BIT}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=HF_TOKEN,
    )

    # ── dtype fix: use `dtype` instead of deprecated `torch_dtype` ──
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    if USE_4BIT and DEVICE == "cuda":
        # 4-bit quantisation — drastically reduces VRAM for large models
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=HF_TOKEN,
            quantization_config=bnb_config,
            device_map="auto",
        )
    elif DEVICE == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=HF_TOKEN,
            dtype=dtype,
            device_map="auto",
        )
    else:
        # CPU path — low_cpu_mem_usage loads weights shard-by-shard,
        # preventing the OOM / segfault seen with large models on low RAM
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=HF_TOKEN,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = model.to(DEVICE)

    model.eval()
    print("Model loaded successfully.\n")
    return tokenizer, model


def generate_text(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate text given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def pipeline_generate(prompt: str, model_id: str = MODEL_ID) -> str:
    """Simpler pipeline-based generation (good for quick tests)."""
    generator = pipeline(
        "text-generation",
        model=model_id,
        device=0 if DEVICE == "cuda" else -1,
        token=HF_TOKEN,
    )
    result = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    return result[0]["generated_text"]


if __name__ == "__main__":
    # ── Option 1: Full manual load ──
    tokenizer, model = load_model(MODEL_ID)
 
    prompt = "The future of artificial intelligence is"
    print(f"Prompt  : {prompt}")
    response = generate_text(prompt, tokenizer, model)
    print(f"Response: {response}\n")