# shared/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    # --- universal
    log_level: str = Field("INFO", env="LOG_LEVEL")
    google_client_id: str = Field(..., env="GOOGLE_CLIENT_ID")
    google_client_secret: str = Field(..., env="GOOGLE_CLIENT_SECRET")
    enc_key: str = Field(..., env="ENC_KEY")
    backend_url: str = Field(..., env="BACKEND_URL")

    # --- groq
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field("openai/gpt-oss-120b", env="GROQ_MODEL")

    # --- backend only
    google_project_id: str = Field("", env="GOOGLE_PROJECT_ID")
    pubsub_topic_name: str = Field("", env="PUBSUB_TOPIC_NAME")
    ws_auth_token: str = Field(..., env="WS_AUTH_TOKEN")

    @validator("enc_key")
    def _check_key(cls, v):
        if len(v) != 44:
            raise ValueError("ENC_KEY must be 32-byte base64")
        return v

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()