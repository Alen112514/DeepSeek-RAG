from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.resolve()
ASSETS_DIR = BASE_DIR / "assets"
HTML_DIR = ASSETS_DIR / "html"
IMAGES_DIR = ASSETS_DIR / "images"
# API Settings
CORS_SETTINGS = {
    "allow_origins": ["*"],  # Update with specific origins for production
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

# Model Settings
LLM_SETTINGS = {
    "model": "deepseek-chat",
    "temperature": 0.8,
    "max_tokens": 512,
    "streaming": True,
}

# API Settings
MULTIMODAL_SETTINGS = {
    "max_text_length": 4096,
    "supported_image_types": ["image/jpeg", "image/png", "image/gif"],
    "max_image_size": 10 * 1024 * 1024  # 10MB
}