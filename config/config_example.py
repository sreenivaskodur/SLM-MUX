"""
Configuration file for SLM-MUX experiments

Copy this file to config.py and fill in your API keys and settings.
"""

# API Configuration
TOGETHER_API_KEY = "your_together_api_key_here"
OPENAI_API_KEY = "your_openai_api_key_here"  # Optional, for verification

# Model Configuration
DEFAULT_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
]

# Inference Configuration
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 2048
DEFAULT_EXTRA_CALLS = 3  # Number of repeated calls per model

# Paths
DATA_DIR = "./data"
OUTPUT_DIR = "./results"

# Concurrency
MAX_WORKERS = 8  # Maximum number of concurrent API calls

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
