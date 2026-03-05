"""
LLM 多 Provider 服务
- 支持 Gemini（云端）/ Ollama（本地）/ 自定义 URL 切换
- 统一流式输出接口 + 合规拦截层
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, AsyncGenerator, Dict, List
import asyncio
import base64
import logging

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── 合规系统 Prompt ────────────────────────────────────────────────
SYSTEM_PROMPT = """你是"宠物阿福"APP内的宠物健康AI助手，定位为"兽医学术文献智能检索工具"，而非执业兽医。

你必须严格遵守以下规则：
1. **禁止使用任何确诊性语言**：绝对禁止"确诊为XX"、"患有XX病"、"诊断结果为"等确定性表述。
2. **必须使用参考性描述**：将所有可能的诊断替换为"具有XX疾病的部分特征"、"高度疑似XX问题"、"症状与XX常见表现相符"。
3. **每次回复结尾必须附加免责声明**：
   "⚠️ 本内容来源于兽医学术文献检索，仅供参考，不作为最终医疗判断依据。请尽快前往正规宠物医院，由执业兽医进行专业诊断。"
4. **识别高危症状**：若用户描述含有"大量出血"、"持续抽搐"、"昏迷"、"呼吸困难"等字样，立即建议紧急就医，不进行病情分析。
5. **语言风格**：用普通主人能理解的通俗语言，避免过度专业术语。结构清晰，使用列表。
"""

# 适配小模型的精简版提示词（减少 token 消耗）
SYSTEM_PROMPT_COMPACT = """你是宠物健康AI助手"宠物阿福"。
规则：1)禁止确诊性语言，用"疑似""可能"替代 2)结尾加免责声明 3)高危症状直接建议就医 4)通俗易懂
"""

EMERGENCY_KEYWORDS = ["大量出血", "持续抽搐", "抽搐不止", "昏迷", "呼吸困难", "休克", "瞳孔散大"]
DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **免责声明**：本内容来源于兽医学术文献检索，仅供参考，不作为最终医疗判断依据。"
    "请尽快前往正规宠物医院，由执业兽医进行专业诊断。"
)


def _is_emergency(text: str) -> bool:
    return any(kw in text for kw in EMERGENCY_KEYWORDS)


def _compliance_check(text: str) -> str:
    replacements = {
        "确诊为": "具有以下疾病部分特征：",
        "确诊": "高度疑似",
        "诊断为": "症状与以下情况相符：",
        "患有": "出现类似",
        "感染了": "疑似存在",
        "必须服用": "建议在兽医指导下考虑",
        "立即注射": "建议尽快就医评估是否需要注射",
    }
    for banned, replacement in replacements.items():
        text = text.replace(banned, replacement)
    return text


# ═══════════════════════════════════════════════════════════════════
#  Provider 抽象基类
# ═══════════════════════════════════════════════════════════════════

class LLMProvider(ABC):
    """所有 LLM Provider 必须实现的接口"""

    name: str = "base"
    display_name: str = "Base Provider"
    supports_image: bool = False

    @abstractmethod
    async def generate_stream(
        self,
        user_text: str,
        system_prompt: str,
        image_base64: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """流式生成文本，yield 每个文本片段"""
        yield ""  # pragma: no cover

    async def health_check(self) -> bool:
        """检查 Provider 是否可用"""
        return True

    def info(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "supports_image": self.supports_image,
        }


# ═══════════════════════════════════════════════════════════════════
#  Gemini Provider（云端）
# ═══════════════════════════════════════════════════════════════════

class GeminiProvider(LLMProvider):
    name = "gemini"
    display_name = "Gemini 2.0 Flash (云端)"
    supports_image = True

    def __init__(self):
        from google import genai
        self._client = genai.Client(api_key=settings.openai_api_key)
        self.model = settings.llm_model or "gemini-2.0-flash"

    async def generate_stream(
        self,
        user_text: str,
        system_prompt: str,
        image_base64: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        from google.genai import types

        contents = []
        if image_base64:
            contents.append(
                types.Part.from_bytes(
                    data=base64.b64decode(image_base64),
                    mime_type="image/jpeg",
                )
            )
        contents.append(user_text)

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=1500,
            temperature=0.3,
        )

        loop = asyncio.get_event_loop()

        def _sync_stream():
            chunks = []
            for chunk in self._client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            ):
                chunks.append(chunk.text or "")
            return chunks

        chunks = await asyncio.wait_for(
            loop.run_in_executor(None, _sync_stream),
            timeout=30.0,
        )
        for text in chunks:
            if text:
                yield text

    async def health_check(self) -> bool:
        try:
            return bool(self._client)
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════
#  Ollama / llama.cpp / OpenAI 兼容 Provider（本地）
# ═══════════════════════════════════════════════════════════════════

class OllamaProvider(LLMProvider):
    """
    支持任何 OpenAI Chat Completions 兼容的本地服务：
    - Ollama (默认 http://localhost:11434/v1)
    - llama.cpp server (--port 8080)
    - 旧手机上的推理服务
    """
    name = "ollama"
    display_name = "本地模型 (Ollama/llama.cpp)"
    supports_image = False  # 小模型一般不支持多模态

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.display_name = f"本地模型 ({self.model})"

    async def generate_stream(
        self,
        user_text: str,
        system_prompt: str,
        image_base64: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "stream": True,
            "max_tokens": 1024,
            "temperature": 0.3,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        return
                    try:
                        import json
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url.replace('/v1', '')}/api/tags")
                return resp.status_code == 200
        except Exception:
            # 尝试 llama.cpp 的 health 接口
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{self.base_url.replace('/v1', '')}/health")
                    return resp.status_code == 200
            except Exception:
                return False

    def info(self) -> dict:
        return {
            **super().info(),
            "base_url": self.base_url,
            "model": self.model,
        }


# ═══════════════════════════════════════════════════════════════════
#  Provider Manager（管理和切换 Provider）
# ═══════════════════════════════════════════════════════════════════

class ProviderManager:
    def __init__(self):
        self._providers: Dict[str, LLMProvider] = {}
        self._active: Optional[str] = None
        self._init_providers()

    def _init_providers(self):
        """根据配置初始化可用的 Provider"""
        # Gemini（如果有 API Key）
        if settings.openai_api_key and settings.openai_api_key != "sk-placeholder":
            try:
                self._providers["gemini"] = GeminiProvider()
                logger.info("✅ Gemini Provider 已加载")
            except Exception as e:
                logger.warning(f"⚠️ Gemini Provider 加载失败: {e}")

        # Ollama / 本地模型（如果配置了）
        if settings.ollama_base_url:
            self._providers["ollama"] = OllamaProvider()
            logger.info(f"✅ Ollama Provider 已加载 ({settings.ollama_model})")

        # 设置默认 Provider
        default = settings.llm_provider
        if default in self._providers:
            self._active = default
        elif self._providers:
            self._active = next(iter(self._providers))

        if self._active:
            logger.info(f"🤖 当前 AI Provider: {self._active}")
        else:
            logger.warning("⚠️ 没有可用的 AI Provider")

    @property
    def active_provider(self) -> Optional[LLMProvider]:
        if self._active:
            return self._providers.get(self._active)
        return None

    def switch(self, provider_name: str) -> bool:
        if provider_name in self._providers:
            self._active = provider_name
            logger.info(f"🔄 AI Provider 切换为: {provider_name}")
            return True
        return False

    def add_provider(self, name: str, provider: LLMProvider):
        """动态添加 Provider（如 App 内置模型连接）"""
        self._providers[name] = provider
        logger.info(f"➕ 新增 AI Provider: {name}")

    def list_providers(self) -> List[dict]:
        result = []
        for name, p in self._providers.items():
            info = p.info()
            info["active"] = name == self._active
            result.append(info)
        return result

    async def check_all_health(self) -> Dict[str, bool]:
        results = {}
        for name, p in self._providers.items():
            results[name] = await p.health_check()
        return results


# ═══════════════════════════════════════════════════════════════════
#  LLM Service（对外统一接口，保持兼容）
# ═══════════════════════════════════════════════════════════════════

class LLMService:
    def __init__(self):
        self.manager = ProviderManager()

    async def analyze_stream(
        self,
        user_text: str,
        image_base64: Optional[str] = None,
        rag_context: Optional[str] = None,
        provider_name: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """统一的流式问诊接口"""
        # 紧急情况直接短路
        if _is_emergency(user_text):
            yield "🚨 **紧急提示**：您描述的症状属于高危状况，请立即停止任何居家处理，"
            yield "**马上前往最近的24小时宠物急诊医院**。时间就是生命，请勿拖延！"
            return

        # 获取 Provider
        if provider_name:
            self.manager.switch(provider_name)
        provider = self.manager.active_provider

        if not provider:
            yield "⚠️ 没有可用的 AI 模型，请在设置中配置模型。"
            return

        # 根据 provider 类型选择提示词
        if isinstance(provider, OllamaProvider):
            system_prompt = SYSTEM_PROMPT_COMPACT  # 小模型用精简版
        else:
            system_prompt = SYSTEM_PROMPT

        # 构建文字（含 RAG 上下文）
        prompt_text = user_text
        if rag_context:
            prompt_text = (
                f"【相关兽医文献参考】\n{rag_context}\n\n"
                f"【用户描述的症状】\n{user_text}"
            )

        # 图片 fallback：小模型不支持图片时给出提示
        if image_base64 and not provider.supports_image:
            yield "📷 当前本地模型不支持图片分析，已仅使用文字进行分析。\n\n"
            image_base64 = None

        full_response = ""
        try:
            async for chunk in provider.generate_stream(prompt_text, system_prompt, image_base64):
                if chunk:
                    cleaned = _compliance_check(chunk)
                    full_response += cleaned
                    yield cleaned

        except asyncio.TimeoutError:
            yield "\n\n⏳ **AI 响应超时**，请稍后重试。"
            return
        except httpx.ConnectError:
            yield "\n\n⚠️ **无法连接本地模型服务**，请检查 Ollama/llama.cpp 是否已启动。"
            return
        except Exception as e:
            logger.error(f"LLM调用失败 [{provider.name}]: {e}")
            yield f"\n\n⚠️ **{provider.display_name} 错误**：{str(e)[:100]}"
            return

        if "免责声明" not in full_response and "仅供参考" not in full_response:
            yield DISCLAIMER


# 单例
llm_service = LLMService()
