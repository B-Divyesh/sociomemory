"""
Configuration management for SocioMemory
"""
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/sociomemory"

    # Redis (optional)
    redis_url: Optional[str] = None

    # OpenAI / Azure OpenAI
    openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_key: Optional[str] = None
    azure_openai_embedding_deployment: str = "text-embedding-ada-002"
    azure_openai_chat_deployment: str = "gpt-4o"  # GPT-4o for high-quality responses

    # Embedding configuration
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072  # text-embedding-3-large has 3072 dims

    # FSRS Configuration
    fsrs_desired_retention: float = 0.9
    fsrs_max_interval: int = 365

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    api_debug: bool = False
    version: str = "0.1.0"
    environment: str = "development"

    # Azure App Service compatibility
    # Azure sets PORT env var, we also check WEBSITES_PORT
    port_override: Optional[int] = None  # Set via PORT env var

    # Security
    api_key: Optional[str] = None
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string"""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def port(self) -> int:
        """Get port with Azure App Service override support"""
        import os
        # Azure App Service sets PORT env var
        azure_port = os.environ.get("PORT") or os.environ.get("WEBSITES_PORT")
        if azure_port:
            return int(azure_port)
        return self.api_port

    @property
    def use_azure_openai(self) -> bool:
        """Check if Azure OpenAI should be used"""
        return bool(self.azure_openai_endpoint and self.azure_openai_key)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
