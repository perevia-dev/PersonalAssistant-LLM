"""Task-scoped OpenAI-compatible model routes with bounded retry and failover.

Model endpoints, order and retry policies live in model_routes.json; individual
route credentials stay in the local .env.local file. Nothing here sends an
extra model request when the old single-model configuration is in use.
"""
from __future__ import annotations

import json
import os
import hashlib
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from .providers import (
    ROOT, ENV_FILE, LlmSettings, ModelRequestError, chat, load_key,
)
from zhuangfy.observability.activity_log import log_event

ROUTES_FILE = ROOT / "model_routes.json"
STAGES = ("planner", "reply", "vision")
STAGE_NAMES = {"planner": "思考", "reply": "回复", "vision": "识图"}
IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
MAX_ROUTES = 5


@dataclass(frozen=True)
class ModelCandidate:
    id: str
    provider: str
    base_url: str
    model: str
    enabled: bool = True
    thinking: str = "auto"
    planner_format: str = "auto"  # auto/prompt/json_object/json_schema

    def validated(self) -> "ModelCandidate":
        if not IDENTIFIER.fullmatch(self.id):
            raise ValueError("模型候选 ID 必须是小写英文字母、数字或下划线")
        if type(self.enabled) is not bool:
            raise ValueError("模型候选的启用设置须为布尔值")
        if self.planner_format not in {"auto", "prompt", "json_object", "json_schema"}:
            raise ValueError("Planner structured format invalid")
        if self.thinking not in {"auto", "enabled", "disabled"}:
            raise ValueError("模型候选的思考设置无效")
        normalized = LlmSettings(self.provider, self.base_url, self.model).validated()
        return replace(self, base_url=normalized.base_url, model=normalized.model)


@dataclass(frozen=True)
class StagePolicy:
    candidates: tuple[ModelCandidate, ...] = ()
    retries: int = 1            # Additional attempts PER candidate.
    retry_delay: float = 0.6     # Seconds, exponential backoff with max 8s.
    timeout: float = 25.0        # Per API request.
    total_timeout: float = 85.0  # Whole stage budget, including backoff.

    def validated(self) -> "StagePolicy":
        if len(self.candidates) > MAX_ROUTES:
            raise ValueError(f"每个阶段最多设置 {MAX_ROUTES} 个模型候选")
        if type(self.retries) is not int or not 0 <= self.retries <= 3:
            raise ValueError("重试次数必须在 0–3 之间")
        for name, value, low, high in (("重试间隔", self.retry_delay, 0, 10),
                                       ("单次请求超时", self.timeout, 3, 120),
                                       ("阶段总超时", self.total_timeout, 5, 240)):
            if type(value) not in (int, float) or not low <= value <= high:
                raise ValueError(f"{name}无效")
        seen = set()
        for item in self.candidates:
            item.validated()
            if item.id in seen:
                raise ValueError("模型候选 ID 不能重复")
            seen.add(item.id)
        return self


@dataclass(frozen=True)
class ModelRoutes:
    planner: StagePolicy = field(default_factory=lambda: StagePolicy(timeout=12, total_timeout=35))
    reply: StagePolicy = field(default_factory=StagePolicy)
    vision: StagePolicy = field(default_factory=StagePolicy)

    def for_stage(self, stage: str) -> StagePolicy:
        if stage not in STAGES:
            raise ValueError("未知模型阶段")
        return getattr(self, stage)

    def validated(self) -> "ModelRoutes":
        used: dict[str, ModelCandidate] = {}
        for stage in STAGES:
            policy = self.for_stage(stage).validated()
            for item in policy.candidates:
                if item.id in used and used[item.id] != item:
                    raise ValueError("同一 API 编号在各阶段必须指向相同供应商和模型")
                used[item.id] = item
        return self


class ModelRoutesFailed(RuntimeError):
    """No configured candidate produced a usable answer."""

    def __init__(self, stage: str, attempts: list[tuple[str, str]]):
        self.stage = stage
        self.attempts = tuple(attempts)
        # Deliberately do not expose endpoints, response bodies or credentials.
        summary = "；".join(f"{model}：{reason}" for model, reason in attempts[-5:])
        super().__init__(f"{STAGE_NAMES[stage]}阶段全部候选服务失败（{len(attempts)} 次尝试）。{summary}")


def _read_policy(raw: object, stage: str) -> StagePolicy:
    if not isinstance(raw, dict):
        raise ValueError(f"{stage} 必须是对象")
    entries = raw.get("candidates", [])
    if not isinstance(entries, list):
        raise ValueError(f"{stage}.candidates 必须是数组")
    candidates = []
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("候选模型必须是对象")
        required = ("id", "provider", "base_url", "model")
        if any(type(item.get(name)) is not str for name in required):
            raise ValueError("候选模型字段格式错误")
        candidates.append(ModelCandidate(
            item["id"], item["provider"], item["base_url"], item["model"],
            item.get("enabled", True), item.get("thinking", "auto"),
            item.get("planner_format", "auto")).validated())
    fallback = getattr(ModelRoutes(), stage)
    kwargs = {field: raw.get(field, getattr(fallback, field))
              for field in ("retries", "retry_delay", "timeout", "total_timeout")}
    return StagePolicy(tuple(candidates), **kwargs).validated()


def load_routes(path: Path = ROUTES_FILE) -> ModelRoutes:
    """Missing file retains old one-model behavior; invalid file reports error."""
    if not path.is_file():
        return ModelRoutes()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("模型路由配置版本错误")
    return ModelRoutes(**{stage: _read_policy(data.get(stage, {}), stage)
                          for stage in STAGES}).validated()


def save_routes(config: ModelRoutes, path: Path = ROUTES_FILE) -> None:
    config.validated()
    raw = {"version": 1}
    for stage in STAGES:
        policy = config.for_stage(stage)
        raw[stage] = {
            "candidates": [dict(id=item.id, provider=item.provider, base_url=item.base_url,
                                model=item.model, enabled=item.enabled, thinking=item.thinking,
                                planner_format=item.planner_format)
                           for item in policy.candidates],
            "retries": policy.retries,
            "retry_delay": policy.retry_delay,
            "timeout": policy.timeout,
            "total_timeout": policy.total_timeout,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def route_secret_name(route_id: str) -> str:
    if not IDENTIFIER.fullmatch(route_id):
        raise ValueError("无效的模型候选 ID")
    return "ZHUANGFY_ROUTE_" + route_id.upper() + "_API_KEY"


def _route_target(provider: str, base_url: str) -> str:
    """Bind private overrides to a specific provider+endpoint, not only a slot."""
    if not isinstance(provider, str) or not isinstance(base_url, str):
        raise ValueError("API 供应商及地址必须是字符串")
    address = base_url.strip().rstrip("/")
    source = (provider + "\0" + address).encode("utf-8")
    return hashlib.sha256(source).hexdigest()[:32]


def load_route_key(route_id: str, provider: str, base_url: str) -> str:
    name = route_secret_name(route_id)
    stored = {}
    try:
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep:
                stored[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    # A model-management provider can supply one key to several model IDs.
    # Prefer an explicitly configured provider connection to imported legacy
    # per-route secrets, so editing its key changes the active credential.
    try:
        from .model_registry import REGISTRY_FILE, load_provider_key, load_registry
        if REGISTRY_FILE.is_file():
            registry = load_registry()
            model = next((item for item in registry.models if item.id == route_id), None)
            connection = next((item for item in registry.providers
                               if model is not None and item.id == model.provider_id), None)
            if connection and (connection.kind, connection.base_url) == (provider, base_url.rstrip("/")):
                provider_key = load_provider_key(connection.id, provider, base_url)
                if provider_key:
                    return provider_key
    except (OSError, ValueError, TypeError, KeyError):
        pass
    # Legacy per-route overrides remain valid for their original endpoint.
    if stored.get(name + "_TARGET") == _route_target(provider, base_url):
        return stored.get(name, "")
    # Official providers have fixed validated URLs, so old provider secrets
    # remain usable. Custom routes must explicitly save their own secrets
    # (but can legitimately be keyless local services).
    return "" if provider == "custom" else load_key(provider)


def save_route_key(route_id: str, secret: str, provider: str, base_url: str) -> None:
    name = route_secret_name(route_id)
    value = secret.strip()
    if not value or "\r" in value or "\n" in value:
        raise ValueError("API Key 不能为空且不能包含换行")
    target = _route_target(provider, base_url)
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    lines = [line for line in lines
             if line.partition("=")[0].strip() not in {name, name + "_TARGET"}]
    lines.extend((name + "=" + value, name + "_TARGET=" + target))
    temporary = ENV_FILE.with_suffix(".tmp")
    try:
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, ENV_FILE)
    finally:
        temporary.unlink(missing_ok=True)


def _reason(error: Exception) -> str:
    # A stable, short code for UI and logs, not arbitrary service error bodies.
    status = getattr(error, "status", None)
    if isinstance(status, int):
        return f"HTTP {status}"
    kind = getattr(error, "kind", "")
    if kind == "timeout":
        return "超时"
    if kind == "network":
        return "网络错误"
    if kind == "empty":
        return "空回复"
    if kind == "protocol":
        return "响应格式错误"
    if isinstance(error, (ValueError, TypeError)):
        return "配置错误"
    return "请求失败"


def _retryable(error: Exception) -> bool:
    return bool(getattr(error, "retryable", False))


def can_attempt_stage(stage: str, settings: LlmSettings, legacy_key: str) -> bool:
    """Return true if any enabled route is callable with current credentials.

    Local custom endpoints can legitimately be keyless.
    """
    policy = load_routes().for_stage(stage)
    active = [item for item in policy.candidates if item.enabled]
    if not active:
        if stage == "vision":
            return False
        return not policy.candidates and (bool(legacy_key) or settings.provider == "custom")
    return any(item.provider == "custom" or bool(load_route_key(item.id, item.provider, item.base_url))
               for item in active)


def invoke_stage(
    stage: str, settings: LlmSettings, legacy_key: str,
    history: list[dict[str, str]], user_text: str, *,
    routes: ModelRoutes | None = None,
    system_prompt: str | None = None,
    legacy_model: str = "",
    memories: list[dict] | None = None,
    emotion_context: str = "",
    context_notes: str = "",
    tool_context: str = "",
    visual_context: str = "",
    trace_id: str = "",
    on_attempt: Callable[[str, int, str], None] | None = None,
    response_validator: Callable[[str], bool] | None = None,
    image_data_url: str | None = None,
    planner_schema: dict | None = None,
) -> str:
    """Try the stage's ordered endpoints. Only retries generation, never delivery.

    All fallback candidates receive the SAME prompt/history. If every service
    fails, raise ModelRoutesFailed; callers decide whether to show an error or
    (for Planner only) proceed with a default plan.
    """
    if planner_schema is not None and (stage != "planner" or not isinstance(planner_schema, dict)):
        raise ValueError("Structured schema only allowed for Planner")
    if stage not in STAGES:
        raise ValueError("未知模型阶段")
    if (stage == "vision") != (image_data_url is not None):
        raise ValueError("识图阶段必须携带图片，其他阶段不能携带图片")
    config = routes if routes is not None else load_routes()
    policy = config.for_stage(stage).validated()
    candidates = [item for item in policy.candidates if item.enabled]
    if not candidates:
        if policy.candidates:
            raise ModelRoutesFailed(stage, [("配置", "没有启用的模型候选")])
        if stage == "vision":
            raise ModelRoutesFailed(stage, [("配置", "尚未配置识图模型")])
        # Migration path: no new routes = existing provider/model and key.
        candidates = [ModelCandidate(
            f"{stage}_legacy", settings.provider, settings.base_url,
            legacy_model or settings.model)]
    deadline = time.monotonic() + policy.total_timeout
    failures: list[tuple[str, str]] = []
    label = STAGE_NAMES[stage]
    for candidate_index, item in enumerate(candidates):
        selected_settings = replace(
            settings, provider=item.provider, base_url=item.base_url,
            model=item.model, system_prompt=system_prompt if system_prompt is not None else settings.system_prompt,
        ).validated()
        key = legacy_key if item.id.endswith("_legacy") else load_route_key(item.id, item.provider, item.base_url)
        for attempt in range(policy.retries + 1):
            remaining = deadline - time.monotonic()
            if remaining < 1:
                failures.append(("阶段总时限", "已用尽"))
                break
            timeout = min(float(policy.timeout), remaining)
            if on_attempt:
                on_attempt(item.id, attempt + 1, label)
            log_event("模型路由", "尝试",
                      f"{trace_id} · {label} · {item.id} · {item.model} · 第 {attempt+1} 次")
            try:
                result = chat(
                    selected_settings, key, history, user_text, memories=memories,
                    emotion_context=emotion_context, context_notes=context_notes,
                    tool_context=tool_context,
                    timeout=timeout, stage=label,
                    **({"thinking": item.thinking} if item.thinking != "auto" else {}),
                    **({"force_json": True} if stage == "planner"
                       and item.planner_format != "prompt" and (
                           planner_schema is not None and (
                               item.planner_format == "json_object"
                               or (item.planner_format == "auto" and
                                   selected_settings.provider == "siliconflow"))
                           or planner_schema is None and candidate_index > 0
                           and selected_settings.provider == "siliconflow") else {}),
                    **({"structured_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "planner_decision_v3",
                                        "strict": True, "schema": planner_schema},
                    }} if stage == "planner" and planner_schema is not None
                    and item.planner_format == "json_schema" else
                    {"structured_format": {"type": "json_object"}}
                    if stage == "planner" and planner_schema is not None
                    and item.planner_format == "json_object" else {}),
                    **({"visual_context": visual_context} if visual_context else {}),
                    **({"image_data_url": image_data_url} if image_data_url is not None else {}))
                if response_validator is not None and not response_validator(result):
                    if stage == "planner":
                        # Retry once as a continuation of this exact exchange.
                        # Repeating the original prompt often makes a prose-first
                        # model repeat prose; an explicit correction is more useful.
                        remaining = deadline - time.monotonic()
                        if remaining >= 1:
                            log_event("模型路由", "格式补救",
                                      f"{trace_id} · {label} · {item.id} · 要求仅返回规划 JSON")
                            correction_history = history[-10:] + [
                                {"role": "user", "content": user_text[:1500]},
                                {"role": "assistant", "content": result[:1200]},
                            ]
                            repaired = chat(
                                selected_settings, key, correction_history,
                                "上一条不是有效的规划 JSON。请依据刚才的用户消息和系统给定字段，"
                                "重新输出且只输出一个 JSON 对象；必须包含 action、confidence、"
                                "delivery，不要自然语言回答、解释或 Markdown。",
                                emotion_context=emotion_context,
                                context_notes=context_notes,
                                timeout=min(float(policy.timeout), remaining), stage=label,
                                force_json=selected_settings.provider == "siliconflow"
                                and item.planner_format != "prompt",
                                **({"structured_format": {
                                    "type": "json_schema",
                                    "json_schema": {"name": "planner_decision_v3",
                                                    "strict": True, "schema": planner_schema},
                                }} if planner_schema is not None
                                and item.planner_format == "json_schema" else
                                {"structured_format": {"type": "json_object"}}
                                if planner_schema is not None and
                                item.planner_format == "json_object" else {}),
                                **({"thinking": item.thinking} if item.thinking != "auto" else {}),
                                **({"visual_context": visual_context} if visual_context else {}))
                            if response_validator(repaired):
                                log_event("模型路由", "格式补救成功",
                                          f"{trace_id} · {label} · {item.id}")
                                log_event("模型路由", "成功", f"{trace_id} · {label} · {item.id}")
                                return repaired
                    raise ModelRequestError(
                        "模型输出格式不符合本阶段要求", kind="protocol", retryable=True)
                log_event("模型路由", "成功", f"{trace_id} · {label} · {item.id}")
                return result
            except Exception as error:
                reason = _reason(error)
                failures.append((item.id, reason))
                log_event("模型路由", "失败",
                          f"{trace_id} · {label} · {item.id} · {reason} · 第 {attempt+1} 次")
                if not _retryable(error) or attempt >= policy.retries:
                    break
                remaining = deadline - time.monotonic()
                delay = min(8.0, policy.retry_delay * (2 ** attempt), max(0.0, remaining - 1.0))
                if delay:
                    time.sleep(delay)
        if time.monotonic() >= deadline:
            break
    raise ModelRoutesFailed(stage, failures)
