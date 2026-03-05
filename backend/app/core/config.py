from __future__ import annotations
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # AI 大模型 (Gemini via OpenAI-compatible)
    openai_api_key: str = "sk-placeholder"
    openai_api_base: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    llm_model: str = "gemini-2.0-flash"

    # LLM Provider 切换（gemini / ollama / local）
    llm_provider: str = "gemini"

    # Ollama / llama.cpp（本地模型）
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen3.5:0.8b"

    # 微信小程序
    wx_appid: str = ""
    wx_secret: str = ""

    # 数据库 (Supabase)
    database_url: str = (
        "postgresql+asyncpg://petafu:petafu123@localhost:5432/petafu_db"
    )

    # Supabase
    supabase_url: str = ""
    supabase_publishable_key: str = ""

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"

    # JWT
    secret_key: str = "dev-secret-key-please-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 43200

    # 知识库
    knowledge_base_dir: str = "../knowledge_base"

    # 管理员账号
    admin_username: str = "admin"
    admin_password: str = "PetAfu@2026!"

    # 合规拦截 - 禁止使用的词汇
    banned_medical_terms: List[str] = [
        "确诊",
        "诊断为",
        "患有",
        "感染了",
        "必须服用",
        "立即注射",
    ]


@lru_cache
def get_settings() -> Settings:
    return Settings()

