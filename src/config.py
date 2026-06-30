from pydantic import BaseModel, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.schemas.constants import (
    CHROMA_DEFAULT_PERSIST_DIR,
    LLM_DEFAULT_TEMPERATURE,
    LLM_PROVIDER_OLLAMA,
    LLM_PROVIDER_OPENAI,
    OLLAMA_DEFAULT_BASE_URL,
    OLLAMA_DEFAULT_MODEL,
    OPENAI_DEFAULT_API_BASE,
    OPENAI_DEFAULT_MODEL,
)


class ChromaConfig(BaseModel):
    persist_directory: str


class LLMConfig(BaseModel):
    provider: str
    base_url: str
    model: str
    temperature: float
    api_key: SecretStr = SecretStr("")


class EmbeddingConfig(BaseModel):
    model: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # App
    app_env: str = "dev"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    allowed_origins: list[str] = ["*"]

    # LLM — provider switch
    #   "ollama": uses ChatOllama
    #   "openai": uses ChatOpenAI (OpenAI-compatible API, e.g. StepFun)
    llm_provider: str = LLM_PROVIDER_OPENAI

    # Ollama 配置（llm_provider = "ollama" 时生效）
    ollama_base_url: str = OLLAMA_DEFAULT_BASE_URL
    llm_model: str = OLLAMA_DEFAULT_MODEL
    llm_temperature: float = LLM_DEFAULT_TEMPERATURE

    # OpenAI-compatible 配置（llm_provider = "openai" 时生效）
    openai_api_base: str = OPENAI_DEFAULT_API_BASE
    openai_model: str = OPENAI_DEFAULT_MODEL
    openai_api_key: str = ""

    # Embedding：DEFAULT_EMBEDDING_MODEL
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # Chroma
    chroma_persist_directory: str = CHROMA_DEFAULT_PERSIST_DIR

    @model_validator(mode="after")
    def _check_openai_key(self):
        if self.llm_provider == LLM_PROVIDER_OPENAI and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY 未设置。请在 .env 文件中添加：OPENAI_API_KEY=你的密钥")
        return self

    @property
    def chroma(self) -> ChromaConfig:
        return ChromaConfig(persist_directory=self.chroma_persist_directory)

    @property
    def llm(self) -> LLMConfig:
        if self.llm_provider == LLM_PROVIDER_OPENAI:
            return LLMConfig(
                provider=LLM_PROVIDER_OPENAI,
                base_url=self.openai_api_base,
                model=self.openai_model,
                temperature=self.llm_temperature,
                api_key=SecretStr(self.openai_api_key),
            )
        return LLMConfig(
            provider=LLM_PROVIDER_OLLAMA,
            base_url=self.ollama_base_url,
            model=self.llm_model,
            temperature=self.llm_temperature,
        )

    @property
    def embedding(self) -> EmbeddingConfig:
        return EmbeddingConfig(model=self.embedding_model)


config = Settings()
