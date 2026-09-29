from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.schemas.constants import (
    AUDIT_DB_PATH,
    CHROMA_DEFAULT_PERSIST_DIR,
    CONVERSATION_DB_PATH,
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
    timeout: float = 30.0
    api_key: SecretStr = SecretStr("")
    # ISSUE-14：显式重试与输出预算。langchain-openai 默认 max_retries=2 会把
    # 单 invoke 最坏耗时放大到 3 × timeout（90s）；1 次重试封顶 2 × timeout。
    max_retries: int = 1
    max_tokens: int = 4096
    # 理解+计划合并调用的输出预算（ISSUE-11/14）：JSON 体积小，超预算截断
    # 走既有 JSONDecodeError 回退路径
    plan_max_tokens: int = 1024


class EmbeddingConfig(BaseModel):
    model: str


class LangfuseConfig(BaseModel):
    enabled: bool = False
    host: str = "https://cloud.langfuse.com"
    public_key: str = ""
    secret_key: SecretStr = SecretStr("")
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    capture_content: bool = False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # App
    app_env: Literal["development", "production", "test"] = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    allowed_origins: list[str] = ["*"]
    api_request_timeout_seconds: float = 60.0

    # 答案语义缓存——默认关闭。缓存未绑定会话上下文与知识库版本，且命中路径
    # 绕过会话保存与审计；重新启用需先满足 issues.md 一.1 的绑定条件。
    semantic_cache_enabled: bool = False

    # LLM — provider switch
    #   "ollama": uses ChatOllama
    #   "openai": uses ChatOpenAI (OpenAI-compatible API, e.g. Volcano Ark)
    llm_provider: str = LLM_PROVIDER_OPENAI

    # Ollama 配置（llm_provider = "ollama" 时生效）
    ollama_base_url: str = OLLAMA_DEFAULT_BASE_URL
    llm_model: str = OLLAMA_DEFAULT_MODEL
    llm_temperature: float = LLM_DEFAULT_TEMPERATURE
    llm_timeout_seconds: float = 30.0
    # ISSUE-14：重试次数与输出 token 预算（两个 provider 共用）
    llm_max_retries: int = 1
    llm_max_tokens: int = 4096
    llm_plan_max_tokens: int = 1024

    # OpenAI-compatible 配置（llm_provider = "openai" 时生效）
    openai_api_base: str = OPENAI_DEFAULT_API_BASE
    openai_model: str = OPENAI_DEFAULT_MODEL
    openai_api_key: str = ""

    # Embedding：入库与检索必须使用同一模型，否则向量空间不匹配导致检索失效。
    # PRD §6.1 选型为 BGE-M3，但当前生产数据（28388 chunks）以 bge-small-zh-v1.5 入库。
    # 切换到 M3 需全量重新入库。Chroma collection metadata 已记录模型名，
    # 不一致时 vector_retriever 和 embedder 会抛出 RuntimeError 阻止启动。
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # Chroma
    chroma_persist_directory: str = CHROMA_DEFAULT_PERSIST_DIR

    # Audit
    audit_db_path: str = AUDIT_DB_PATH
    conversation_db_path: str = CONVERSATION_DB_PATH

    # Langfuse — Agent/LLM 链路观测，默认关闭。职责边界：只观察链路元数据
    # （耗时、token、模型调用）；权限/引用/合规审计仍在 SQLite，QPS/延迟/
    # 错误率指标仍在 Prometheus。隐私红线：问题原文、模型回答、文档与 chunk
    # 文本默认不进 payload；capture_content 仅供开发调试显式开启并经统一脱敏。
    langfuse_enabled: bool = False
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    langfuse_capture_content: bool = False

    @model_validator(mode="after")
    def _check_openai_key(self):
        if self.llm_provider == LLM_PROVIDER_OPENAI and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY 未设置。请在 .env 文件中配置有效密钥")
        return self

    @model_validator(mode="after")
    def _check_langfuse_config(self):
        if not self.langfuse_enabled:
            return self
        missing = [
            name
            for name, value in (
                ("LANGFUSE_HOST", self.langfuse_host),
                ("LANGFUSE_PUBLIC_KEY", self.langfuse_public_key),
                ("LANGFUSE_SECRET_KEY", self.langfuse_secret_key.get_secret_value()),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"LANGFUSE_ENABLED=true 但 {'、'.join(missing)} 未设置。"
                "请在 .env 文件中配置，或将 LANGFUSE_ENABLED 置为 false"
            )
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
                timeout=self.llm_timeout_seconds,
                api_key=SecretStr(self.openai_api_key),
                max_retries=self.llm_max_retries,
                max_tokens=self.llm_max_tokens,
                plan_max_tokens=self.llm_plan_max_tokens,
            )
        return LLMConfig(
            provider=LLM_PROVIDER_OLLAMA,
            base_url=self.ollama_base_url,
            model=self.llm_model,
            temperature=self.llm_temperature,
            timeout=self.llm_timeout_seconds,
            max_retries=self.llm_max_retries,
            max_tokens=self.llm_max_tokens,
            plan_max_tokens=self.llm_plan_max_tokens,
        )

    @property
    def embedding(self) -> EmbeddingConfig:
        return EmbeddingConfig(model=self.embedding_model)

    @property
    def langfuse(self) -> LangfuseConfig:
        return LangfuseConfig(
            enabled=self.langfuse_enabled,
            host=self.langfuse_host,
            public_key=self.langfuse_public_key,
            secret_key=self.langfuse_secret_key,
            sample_rate=self.langfuse_sample_rate,
            capture_content=self.langfuse_capture_content,
        )


config = Settings()
