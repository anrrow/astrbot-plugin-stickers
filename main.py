"""
sticker_master — AstrBot 表情包 AI 工具插件
============================================
核心理念：LLM Tool Use + 语境门控 + 软兜底 + 人格专属表情包。

本版修复/增强：
1. /sticker_add 支持多行「含义 URL」粘贴，也支持「(人格分组名) 含义 URL」。
2. 增加 /sticker_import，语义同 /sticker_add，更适合批量粘贴 URL。
3. /sticker_list_custom 不再让人误会：会提示 custom.json 与 default.json 的区别。
4. 增加人格专属表情包：/sticker_role 萌猫 后，本会话优先/限定使用萌猫表情包。
5. LLM 工具 send_sticker 会按当前匹配人格分组匹配，不再把不同角色同名 meaning 覆盖掉。
6. 增加原始消息监听兜底：QQ 多行 /sticker_add 被命令解析器吞掉时，也能直接导入。
7. 增加 WebUI 配置面板粘贴栏：别人安装插件后可直接在插件配置中粘贴自己的表情包，不需要改 GitHub。
8. 增加人格分组面板：每个角色一个折叠项，人格分组名 + URL 粘贴栏，便于区分谁是谁。

作者：Anrrow
版本：1.7.1-persona-network-fix
"""

import json
import os
import difflib
import logging
import inspect
import random
import re
import time
import asyncio
import hashlib
from typing import Any

from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import ProviderRequest
import astrbot.api.message_components as Comp

try:
    import aiohttp
except Exception:  # pragma: no cover - AstrBot normally includes aiohttp
    aiohttp = None

# AstrBot 新版推荐 filter.llm_tool；旧版可能只有 astrbot.api.llm_tool，这里做兼容。
try:
    _llm_tool = filter.llm_tool
except AttributeError:  # pragma: no cover - 兼容旧版 AstrBot
    from astrbot.api import llm_tool as _llm_tool

# 动态提示优先塞到 extra_user_content_parts，避免每轮改 system_prompt。
try:
    from astrbot.core.agent.message import TextPart
except Exception:  # pragma: no cover - 兼容旧版 AstrBot
    TextPart = None

logger = logging.getLogger(__name__)

CUSTOM_FILE = "custom.json"
URL_RE = re.compile(r"https?://\S+", re.I)
ROLE_PREFIX_RE = re.compile(r"^\s*[（(]([^()（）]{1,40})[）)]\s*(.*?)\s*$")


@register(
    "sticker_master",
    "Anrrow",
    "AI 驱动表情包工具 - 让 AI 像真人一样主动发送表情包",
    "1.7.1-persona-network-fix",
)
class StickerMaster(Star):
    """
    通过 LLM Tool 让 AI 主动调用 send_sticker()。

    表情包 JSON 支持两种格式：
    1. 普通：{"meaning":"开心", "url":"https://..."}
    2. 人格专属：{"role":"萌猫", "meaning":"哭哭", "url":"https://..."}

    也支持把 role 写在 meaning 开头：
      {"meaning":"(萌猫) 哭哭", "url":"https://..."}

    批量添加命令：
      /sticker_add
      (萌猫) 卧槽尼玛理理我啊 https://files.catbox.moe/a2erwr.jpeg
      小狗哭哭 https://files.catbox.moe/outz25.gif

    人格调试命令：
      /sticker_role 萌猫     # 当前会话启用萌猫专属表情包
      /sticker_role off      # 关闭当前匹配人格分组限定
      /sticker_roles         # 查看有哪些角色
    """

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context, config or {})
        self.config = config or {}

        # 展示/总表：display_meaning -> url。角色表情的 display_meaning 为「(角色) 含义」。
        self.sticker_map: dict[str, str] = {}
        self.meanings_list: list[str] = []

        # 匹配用表：普通表情与人格专属表情分开，避免「哭哭」被不同角色互相覆盖。
        self.common_map: dict[str, str] = {}
        self.role_sticker_map: dict[str, dict[str, str]] = {}

        self.sticker_probability = self._cfg_float("sticker_probability", 0.65, 0.0, 1.0)
        self.sticker_sample_size = self._cfg_int("sticker_sample_size", 50, 5, 200)
        self.fallback_enabled = self._cfg_bool("sticker_fallback_enabled", True)
        self.cooldown_seconds = self._cfg_int("sticker_cooldown_seconds", 45, 0, 3600)
        self.min_score = self._cfg_int("sticker_min_score", 2, 1, 10)
        self.nudge_after_turns = self._cfg_int("sticker_nudge_after_turns", 2, 1, 10)
        self.aggressive_score = self._cfg_int("sticker_aggressive_score", 4, 2, 10)
        self.debug = self._cfg_bool("sticker_debug", False)

        # 人格相关配置。
        # sticker_auto_persona_enabled：开启后，自动读取当前会话正在使用的 AstrBot 人格。
        # sticker_default_role：兜底人格/表情包分组名。留空则不限定。
        # sticker_role_strict：开启后，当前人格只用该人格专属 + 通用表情；关闭则匹配不到时可回退到全库。
        self.auto_persona_enabled = self._cfg_bool("sticker_auto_persona_enabled", True)
        self.default_role = str(self._cfg("sticker_default_role", "") or "").strip()
        self.role_strict = self._cfg_bool("sticker_role_strict", True)
        self.persona_alias_map: dict[str, str] = {}
        self.session_persona_cache: dict[str, str] = {}
        self.session_persona_debug: dict[str, str] = {}

        # WebUI 配置面板导入区：
        # 1) sticker_config_text：一个大粘贴栏，支持通用与 (角色) 前缀。
        # 2) sticker_role_groups：人格分组面板，每个角色一个 role_name + stickers_text。
        # 这些内容保存在 AstrBot 的 data/config/<plugin>_config.json，不写进 GitHub。
        self.config_sticker_text = str(self._cfg("sticker_config_text", "") or "")
        self.config_persona_groups = self._cfg("sticker_persona_groups", []) or []
        self.config_role_groups = self._cfg("sticker_role_groups", []) or []  # 兼容旧配置
        self._last_config_fail_list: list[tuple[str, str]] = []

        self.session_roles: dict[str, str] = {}

        # 用于判断“这一轮是不是 LLM 回复”、避免重复补发，以及控制兜底频率。
        self._recent_llm_event_keys: dict[str, float] = {}
        self._tool_sent_event_keys: dict[str, float] = {}
        self._last_sticker_session_ts: dict[str, float] = {}
        self._eligible_no_sticker_turns: dict[str, int] = {}
        self._admin_command_event_keys: dict[str, float] = {}

        self._reload()

    # ═══════════════════════════════════════════════════════════
    # 配置读取
    # ═══════════════════════════════════════════════════════════

    def _cfg(self, key: str, default: Any) -> Any:
        try:
            if isinstance(self.config, dict):
                return self.config.get(key, default)
            if hasattr(self.config, "get"):
                return self.config.get(key, default)
        except Exception:
            pass
        return default

    def _cfg_bool(self, key: str, default: bool) -> bool:
        value = self._cfg(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "开启", "是"}
        return bool(value)

    def _cfg_int(self, key: str, default: int, min_v: int, max_v: int) -> int:
        try:
            value = int(self._cfg(key, default))
        except Exception:
            value = default
        return max(min_v, min(max_v, value))

    def _cfg_float(self, key: str, default: float, min_v: float, max_v: float) -> float:
        try:
            value = float(self._cfg(key, default))
        except Exception:
            value = default
        return max(min_v, min(max_v, value))

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _norm_key(self, value: str) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[\s\-＿_·•:：,，。.!！?？()（）\[\]【】]+", "", text)
        return text

    def _register_persona_alias(self, alias: str, pack_name: str) -> None:
        alias = str(alias or "").strip()
        pack_name = str(pack_name or "").strip()
        if not alias or not pack_name:
            return
        self.persona_alias_map[self._norm_key(alias)] = pack_name

    # ═══════════════════════════════════════════════════════════
    # 数据加载
    # ═══════════════════════════════════════════════════════════

    def _stickers_dir(self) -> str:
        return os.path.join(os.path.dirname(__file__), "stickers")

    def _display_meaning(self, meaning: str, role: str = "") -> str:
        return f"({role}) {meaning}" if role else meaning

    def _split_role_prefix(self, meaning: str) -> tuple[str, str]:
        """把「(萌猫) 哭哭」拆成 role=萌猫, meaning=哭哭。"""
        raw = (meaning or "").strip()
        m = ROLE_PREFIX_RE.match(raw)
        if not m:
            return "", raw
        role = (m.group(1) or "").strip()
        rest = (m.group(2) or "").strip()
        # 防止只有「(萌猫)」没有实际含义。
        return (role, rest) if role and rest else ("", raw)

    def _normalise_item(self, item: dict) -> tuple[str, str, str]:
        """返回 role, meaning, url。role 可以为空。"""
        raw_meaning = str(item.get("meaning", "") or "").strip()
        url = str(item.get("url", "") or "").strip()
        role = str(
            item.get("role")
            or item.get("character")
            or item.get("char")
            or item.get("owner")
            or item.get("persona")
            or ""
        ).strip()

        prefix_role, clean_meaning = self._split_role_prefix(raw_meaning)
        if prefix_role and not role:
            role = prefix_role
            raw_meaning = clean_meaning

        return role, raw_meaning, url

    def _put_sticker(self, role: str, meaning: str, url: str) -> None:
        role = (role or "").strip()
        meaning = (meaning or "").strip()
        url = (url or "").strip()
        if not meaning or not url:
            return
        if role:
            self.role_sticker_map.setdefault(role, {})[meaning] = url
        else:
            self.common_map[meaning] = url
        self.sticker_map[self._display_meaning(meaning, role)] = url

    def _reload(self) -> None:
        """扫描 stickers/ 目录，加载/热重载所有 .json 表情包文件。"""
        self.sticker_map.clear()
        self.common_map.clear()
        self.role_sticker_map.clear()
        self.persona_alias_map.clear()

        stickers_dir = self._stickers_dir()
        if not os.path.exists(stickers_dir):
            os.makedirs(stickers_dir, exist_ok=True)
            logger.warning(
                "[StickerMaster] ⚠️ stickers/ 目录刚创建，"
                "请把你的表情包 JSON 文件放进去再重载"
            )
            self.meanings_list = []
            return

        total = 0
        for fname in sorted(os.listdir(stickers_dir)):
            if not fname.endswith(".json"):
                continue

            fpath = os.path.join(stickers_dir, fname)
            try:
                items = self._read_sticker_json(fpath)
                n = 0
                for item in items:
                    role, meaning, url = self._normalise_item(item)
                    if meaning and url:
                        self._put_sticker(role, meaning, url)
                        n += 1
                total += n
                logger.info(f"[StickerMaster] 加载 {fname}: {n} 个")
            except Exception as e:
                logger.error(f"[StickerMaster] 加载 {fname} 失败: {e}")

        config_total, config_fail = self._load_config_stickers()
        self._last_config_fail_list = config_fail
        total += config_total

        self.meanings_list = list(self.sticker_map.keys())
        logger.info(
            f"[StickerMaster] ✅ 就绪，共 {total} 个表情包，"
            f"其中配置面板 {config_total} 个，"
            f"角色 {len(self.role_sticker_map)} 个，通用 {len(self.common_map)} 个"
        )
        if config_fail:
            logger.warning(f"[StickerMaster] 配置面板有 {len(config_fail)} 条未解析")

    def _read_sticker_json(self, fpath: str) -> list[dict]:
        """兼容 list 格式，也兼容 {stickers:[...]} / {分类:[...]} 格式。"""
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]

        if isinstance(data, dict):
            if isinstance(data.get("stickers"), list):
                return [x for x in data["stickers"] if isinstance(x, dict)]

            merged: list[dict] = []
            for value in data.values():
                if isinstance(value, list):
                    merged.extend(x for x in value if isinstance(x, dict))
            return merged

        return []

    def _maybe_role_section(self, line: str) -> str:
        """识别配置面板中的角色分段标题。

        支持：
          [萌猫]
          【萌猫】
          # 萌猫
          # 角色：萌猫
          角色：萌猫
          role: 萌猫
          [通用] / # 通用 / common 会切回通用表情。
        """
        t = (line or "").strip()
        if not t:
            return ""

        m = re.match(r"^[\[【]\s*([^\]】]{1,40})\s*[\]】]$", t)
        if m:
            role = m.group(1).strip()
            return "" if role.lower() in {"通用", "common", "global", "default", "none"} else role

        if t.startswith("#"):
            t = t.lstrip("#").strip()
            if not t:
                return "__comment__"

        m = re.match(r"^(?:角色|role|char|character)\s*[:：]\s*(.{1,40})$", t, re.I)
        if m:
            role = m.group(1).strip()
            return "" if role.lower() in {"通用", "common", "global", "default", "none"} else role

        if t.lower() in {"通用", "common", "global", "default", "none"}:
            return ""

        return "__not_section__"

    def _parse_sticker_text_block(
        self,
        body: str,
        default_role: str = "",
    ) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
        """解析 WebUI 粘贴栏，并支持角色分段。"""
        ok_list: list[tuple[str, str, str]] = []
        fail_list: list[tuple[str, str]] = []
        current_role = (default_role or "").strip()

        for raw_line in (body or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("//"):
                continue

            if not URL_RE.search(line):
                section = self._maybe_role_section(line)
                if section == "__comment__":
                    continue
                if section != "__not_section__":
                    current_role = section
                    continue
                continue

            match = URL_RE.search(line)
            if not match:
                continue

            url = match.group(0).strip().rstrip("，,。)")
            meaning_part = line[:match.start()].strip() or line[match.end():].strip()
            if not meaning_part:
                fail_list.append((line, "缺少含义；格式应为：含义 URL"))
                continue

            prefix_role, meaning = self._split_role_prefix(meaning_part)
            role = prefix_role or current_role
            if not meaning:
                fail_list.append((line, "含义为空"))
                continue

            ok_list.append((role, meaning, url))

        return ok_list, fail_list

    def _load_config_stickers(self) -> tuple[int, list[tuple[str, str]]]:
        """从 WebUI 配置面板读取表情包，不写 custom.json。"""
        ok_all: list[tuple[str, str, str]] = []
        fail_all: list[tuple[str, str]] = []

        ok, fail = self._parse_sticker_text_block(self.config_sticker_text)
        ok_all.extend(ok)
        fail_all.extend(fail)

        group_sources: list[Any] = []
        if isinstance(self.config_persona_groups, list):
            group_sources.extend(self.config_persona_groups)
        if isinstance(self.config_role_groups, list):
            group_sources.extend(self.config_role_groups)

        for group in group_sources:
            if not isinstance(group, dict):
                continue
            role_name = str(
                group.get("persona_name")
                or group.get("role_name")
                or group.get("role")
                or group.get("人格名")
                or group.get("人格分组名")
                or ""
            ).strip()
            aliases_raw = str(
                group.get("persona_aliases")
                or group.get("aliases")
                or group.get("别名")
                or ""
            )
            text = str(group.get("stickers_text") or group.get("text") or group.get("表情包") or "")
            if not role_name and not text.strip():
                continue
            if role_name:
                self._register_persona_alias(role_name, role_name)
                for alias in re.split(r"[,，;；\n]+", aliases_raw):
                    self._register_persona_alias(alias, role_name)
            ok, fail = self._parse_sticker_text_block(text, default_role=role_name)
            ok_all.extend(ok)
            fail_all.extend(fail)

        for role, meaning, url in ok_all:
            self._put_sticker(role, meaning, url)

        return len(ok_all), fail_all

    # ═══════════════════════════════════════════════════════════
    # 人格状态：自动把当前 AstrBot 人格映射到表情包分组
    # ═══════════════════════════════════════════════════════════

    def _current_role(self, event: AstrMessageEvent | None = None) -> str:
        """同步兜底：优先手动覆盖，其次使用最近一次自动识别缓存，再其次默认分组。"""
        if event is not None:
            session_key = self._session_key(event)
            session_role = self.session_roles.get(session_key, "")
            if session_role:
                return session_role
            cached = self.session_persona_cache.get(session_key, "")
            if cached:
                return cached
        return self.default_role

    def _match_persona_to_pack(self, candidates: list[str]) -> str:
        """把 AstrBot 人格名/persona_id 映射到表情包分组名。"""
        if not candidates:
            return ""
        for cand in candidates:
            nk = self._norm_key(cand)
            if nk and nk in self.persona_alias_map:
                return self.persona_alias_map[nk]
        role_keys = {self._norm_key(role): role for role in self.role_sticker_map.keys()}
        for cand in candidates:
            nk = self._norm_key(cand)
            if nk and nk in role_keys:
                return role_keys[nk]
        for cand in candidates:
            nk = self._norm_key(cand)
            if not nk:
                continue
            for role_nk, role in role_keys.items():
                if role_nk and (role_nk in nk or nk in role_nk):
                    return role
        return ""

    async def _detect_persona_candidates(self, event: AstrMessageEvent | None) -> list[str]:
        """尽量从 AstrBot 当前会话读取人格名/persona_id。失败时返回空列表，不影响插件运行。"""
        if event is None:
            return []
        candidates: list[str] = []
        umo = getattr(event, "unified_msg_origin", None) or getattr(event, "session_id", None)
        def add(value: Any):
            if value is None:
                return
            if isinstance(value, str):
                v = value.strip()
                if v and v not in candidates:
                    candidates.append(v)
                return
            if isinstance(value, dict):
                for k in ("name", "persona_id", "id", "display_name", "title"):
                    add(value.get(k))
                return
            for attr in ("name", "persona_id", "id", "display_name", "title"):
                try:
                    add(getattr(value, attr, None))
                except Exception:
                    pass
        try:
            conv_mgr = getattr(self.context, "conversation_manager", None)
            if conv_mgr is not None and umo:
                cid = await self._maybe_await(conv_mgr.get_curr_conversation_id(umo))
                if cid:
                    conv = await self._maybe_await(conv_mgr.get_conversation(umo, cid))
                    add(conv)
                    persona_id = getattr(conv, "persona_id", None) if conv is not None else None
                    if persona_id:
                        add(persona_id)
                        persona_mgr = getattr(self.context, "persona_manager", None)
                        if persona_mgr is not None and hasattr(persona_mgr, "get_persona"):
                            try:
                                add(await self._maybe_await(persona_mgr.get_persona(persona_id)))
                            except Exception:
                                pass
        except Exception as e:
            if self.debug:
                logger.debug(f"[StickerMaster] 读取 conversation persona 失败: {e}")
        try:
            persona_mgr = getattr(self.context, "persona_manager", None)
            if persona_mgr is not None and hasattr(persona_mgr, "get_default_persona_v3"):
                add(await self._maybe_await(persona_mgr.get_default_persona_v3(umo)))
        except Exception as e:
            if self.debug:
                logger.debug(f"[StickerMaster] 读取 default persona 失败: {e}")
        return candidates

    async def _resolve_current_role(self, event: AstrMessageEvent | None = None) -> str:
        if event is None:
            return self.default_role
        session_key = self._session_key(event)
        manual = self.session_roles.get(session_key, "")
        if manual:
            return manual
        if self.auto_persona_enabled:
            candidates = await self._detect_persona_candidates(event)
            matched = self._match_persona_to_pack(candidates)
            self.session_persona_debug[session_key] = ", ".join(candidates) if candidates else "未读取到"
            if matched:
                self.session_persona_cache[session_key] = matched
                return matched
            self.session_persona_cache.pop(session_key, None)
        return self.default_role

    def _available_meanings_for_role(self, role: str = "") -> list[str]:
        """给模型看的 meaning 样例。检测到人格后，优先给该人格 + 通用。"""
        role = (role or "").strip()
        if not role:
            return self.meanings_list[:]
        role_items = [self._display_meaning(m, role) for m in self.role_sticker_map.get(role, {}).keys()]
        common_items = list(self.common_map.keys())
        if self.role_strict:
            return role_items + common_items
        other = [m for m in self.meanings_list if m not in set(role_items + common_items)]
        return role_items + common_items + other

    async def _available_meanings(self, event: AstrMessageEvent | None = None) -> list[str]:
        return self._available_meanings_for_role(await self._resolve_current_role(event))

    # ═══════════════════════════════════════════════════════════
    # LLM 请求注入：让模型更愿意调用工具
    # ═══════════════════════════════════════════════════════════

    @filter.on_llm_request()
    async def inject_sticker_hint(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self.meanings_list:
            return

        self._touch_recent(self._recent_llm_event_keys, self._event_key(event))
        self._cleanup_recent()

        current_role = await self._resolve_current_role(event)
        pool = self._available_meanings_for_role(current_role)
        if not pool:
            return
        sample = random.sample(pool, min(self.sticker_sample_size, len(pool)))
        role_line = ""
        if current_role:
            role_line = (
                f"当前 AstrBot 人格已匹配到表情包分组「{current_role}」。"
                "调用 send_sticker 时优先选择这个人格分组的 meaning；不要混用其他人格专属表情包。\n"
            )

        hint = (
            "<sticker_master_runtime_hint>\n"
            "你现在拥有 send_sticker 工具，可以发送真实表情包图片。\n"
            + role_line +
            "核心原则：你自己判断该不该发；适合就主动调用，不适合就不要调用。\n"
            "不要把表情包当成每轮固定流程，也不要因为怕打扰就一直不用。把它当成真人聊天里的自然配图。\n"
            "适合主动调用 send_sticker 的场景：\n"
            "- 轻松闲聊、撒娇、调侃、安慰、打招呼、晚安、庆祝、表达想念/开心/委屈/无语/震惊等明显情绪。\n"
            "- 当你本来想写‘抱抱/摸摸/亲亲/笑死/无语/晚安/好耶’这类情绪动作时，优先考虑真正调用工具。\n"
            "不适合调用 send_sticker 的场景：\n"
            "- 线下模式、现实行动模式、严肃讨论、代码排错、插件配置、论文/文献、合同/法务、数据/表格、长篇分析。\n"
            "- 用户明确说不要发表情包、正在要你专注解决问题、或回复必须极短。\n"
            "频率规则：\n"
            "- 日常轻松聊天可以偏积极使用，但不要连续很多轮都发；每次最多调用 1 次。\n"
            "- 用户只是把你叫作“宝宝/老公/老婆”，不构成发表情包的理由；要看这一轮有没有情绪和聊天氛围。\n"
            "- 如果这一轮明显适合发，不要等插件兜底，你可以主动调用。\n"
            "meaning 尽量从下面样例里精确选择；找不到完全合适的，再自由描述情绪。\n"
            "本轮可用表情包样例：\n"
            + " / ".join(sample) +
            "\n</sticker_master_runtime_hint>"
        )

        # 新版 AstrBot：动态提示放 extra_user_content_parts，并标记临时，避免污染历史。
        if TextPart is not None and hasattr(req, "extra_user_content_parts"):
            try:
                part = TextPart(text=hint)
                if hasattr(part, "mark_as_temp"):
                    part = part.mark_as_temp()
                req.extra_user_content_parts.append(part)
                return
            except Exception as e:
                logger.warning(f"[StickerMaster] extra_user_content_parts 注入失败，回退 system_prompt: {e}")

        # 旧版兼容：没有 extra_user_content_parts 时只能追加 system_prompt。
        req.system_prompt = (req.system_prompt or "") + "\n\n" + hint

    # ═══════════════════════════════════════════════════════════
    # 发送前兜底：模型偷懒没调用工具时，插件自己补一个
    # ═══════════════════════════════════════════════════════════

    @filter.on_decorating_result()
    async def auto_append_sticker_if_needed(self, event: AstrMessageEvent):
        if not self.fallback_enabled or not self.sticker_map:
            return

        msg = (getattr(event, "message_str", "") or "").strip()
        event_key = self._event_key(event)

        if not self._is_recent(self._recent_llm_event_keys, event_key, 120):
            # 不是刚刚经过 LLM 的普通回复，可能是命令结果或其他插件输出，不乱加。
            return

        if self._is_recent(self._tool_sent_event_keys, event_key, 120):
            # LLM 已经主动调用过 send_sticker 了，不重复加。
            return

        try:
            result = event.get_result()
            chain = getattr(result, "chain", None)
            if chain is None:
                return

            if self._chain_has_image(chain):
                return

            reply_text = self._chain_to_text(chain).strip()
            visible_text = msg + "\n" + reply_text

            if not self._context_allows_sticker(msg, reply_text):
                if self.debug:
                    logger.info("[StickerMaster] 兜底跳过：语境不适合发表情包")
                return

            score = self._sticker_moment_score(msg, reply_text)
            if score < self.min_score:
                if self.debug:
                    logger.info(f"[StickerMaster] 兜底跳过：情绪分数不足 score={score}")
                return

            session_key = self._session_key(event)
            if self._in_session_cooldown(session_key):
                if self.debug:
                    logger.info("[StickerMaster] 兜底跳过：兜底冷却中")
                return

            no_sticker_turns = self._eligible_no_sticker_turns.get(session_key, 0) + 1
            self._eligible_no_sticker_turns[session_key] = no_sticker_turns

            should_nudge = score >= self.aggressive_score or no_sticker_turns >= self.nudge_after_turns
            if not should_nudge:
                if self.debug:
                    logger.info(
                        f"[StickerMaster] 兜底跳过：等待 AI 自己决定 "
                        f"score={score}, no_sticker_turns={no_sticker_turns}/{self.nudge_after_turns}"
                    )
                return

            if random.random() > self.sticker_probability:
                if self.debug:
                    logger.info(
                        f"[StickerMaster] 软兜底概率未命中 "
                        f"score={score}, no_sticker_turns={no_sticker_turns}, p={self.sticker_probability}"
                    )
                return

            role = await self._resolve_current_role(event)
            meaning = self._guess_meaning(visible_text, role=role)
            url, matched = self._match(meaning, role=role)
            if not url:
                if self.debug:
                    logger.info(f"[StickerMaster] 兜底跳过：未匹配到 meaning='{meaning}', role='{role}'")
                return

            image_comp = await self._image_component_from_url(url)
            if image_comp is None:
                logger.warning(f"[StickerMaster] 兜底表情包发送取消：图片源下载失败，matched={matched}, url={url}")
                return
            chain.append(image_comp)
            self._touch_recent(self._tool_sent_event_keys, event_key)
            self._mark_session_sticker_sent(session_key)
            logger.info(f"[StickerMaster] 智能兜底补发: role={role or '-'}, score={score}, '{meaning}' → '{matched}' → {url}")
        except Exception as e:
            logger.error(f"[StickerMaster] 兜底补发表情包失败: {e}")

    def _should_skip_by_user_text(self, text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        if text.startswith("/") or text.startswith("sticker_"):
            return True
        skip_words = (
            "不要发表情", "别发表情", "先别发表情", "禁止发表情",
            "不要表情包", "别发图", "别发图片", "严肃模式", "认真点", "别闹",
            "线下模式", "现实模式", "现实行动模式", "offline mode",
        )
        return any(w in lowered for w in skip_words)

    def _context_allows_sticker(self, user_text: str, reply_text: str = "") -> bool:
        user_text = user_text or ""
        reply_text = reply_text or ""
        combined = (user_text + "\n" + reply_text).lower()

        if self._should_skip_by_user_text(user_text):
            return False

        # 长篇说明、代码块、列表教程、结构化输出，发图会打断阅读。
        if len(reply_text) >= 420 or reply_text.count("\n") >= 7:
            return False

        hard_markers = (
            "```", "<code", "</code", "traceback", "exception",
            "function ", "class ", "def ", "async def ", "@filter",
            "{", "}", "=>", "->", "==", "!=",
        )
        if any(m in combined for m in hard_markers):
            return False

        serious_words = (
            # 开发/插件/排错
            "代码", "脚本", "插件", "bug", "报错", "日志", "接口", "api",
            "tokens", "token", "配置", "安装", "重启", "修复", "怎么改", "逻辑",
            "悬浮窗", "安卓", "github", "main.py", "json",
            # 学术/文档/数据
            "论文", "文献", "期刊", "ssci", "c刊", "北核", "综述", "理论",
            "变量", "模型", "假设", "中介", "调节", "数据", "表格", "公式",
            "zotero", "pdf", "word", "ppt", "框架", "生成完整",
            # 合同/法务/金融类
            "合同", "协议", "法务", "律师", "股权", "转让", "质押", "抵押",
            "贷款", "对价", "代持", "条款",
            # 线下/现实模式
            "线下模式", "现实模式", "现实行动", "面对面",
        )
        if any(w in combined for w in serious_words):
            return False

        return True

    def _sticker_moment_score(self, user_text: str, reply_text: str = "") -> int:
        text = (user_text + "\n" + reply_text).lower()
        score = 0

        strong_groups: list[tuple[tuple[str, ...], int]] = [
            (("想你", "好想", "念你"), 3),
            (("爱你", "喜欢你", "亲亲", "贴贴", "抱抱", "摸摸", "撒娇"), 3),
            (("呜呜", "哭哭", "委屈", "难过", "心疼", "大哭"), 3),
            (("哈哈", "笑死", "乐死", "嘿嘿", "嘻嘻", "好耶", "可爱", "开心", "高兴", "快乐"), 2),
            (("晚安", "早安", "早上好", "睡觉", "困了"), 2),
            (("谢谢", "感谢", "辛苦啦", "对不起", "抱歉"), 2),
            (("无语", "离谱", "服了", "气死", "生气", "哼"), 2),
            (("震惊", "真的假的", "不会吧", "啊？", "啊?", "一脸懵"), 2),
            (("饿", "吃饭", "开饭", "美味"), 2),
        ]

        for keywords, weight in strong_groups:
            if any(k in text for k in keywords):
                score += weight

        if len(user_text.strip()) <= 25 and len(reply_text.strip()) <= 120:
            if any(x in text for x in ("～", "！", "!!", "？", "??", "呀", "嘛", "啦", "喔")):
                score += 1

        return score

    # ═══════════════════════════════════════════════════════════
    # 匹配算法
    # ═══════════════════════════════════════════════════════════

    def _match_in_map(self, query: str, mp: dict[str, str], prefix: str = "") -> tuple[str | None, str | None]:
        if not mp:
            return None, None
        q = (query or "").strip()
        if not q:
            return None, None

        if q in mp:
            return mp[q], self._display_meaning(q, prefix)

        for m, u in mp.items():
            if q in m or m in q:
                return u, self._display_meaning(m, prefix)

        hits = difflib.get_close_matches(q, list(mp.keys()), n=1, cutoff=0.28)
        if hits:
            return mp[hits[0]], self._display_meaning(hits[0], prefix)

        return None, None

    def _match(self, query: str, role: str = "") -> tuple[str | None, str | None]:
        """
        精确 → 子串 → difflib 模糊。
        设置 role 后：先查该角色，再查通用；若 sticker_role_strict=false，再查全库。
        query 也可以直接传「(萌猫) 哭哭」。
        """
        if not self.sticker_map:
            return None, None

        q = (query or "").strip()
        if not q:
            return None, None

        query_role, clean_q = self._split_role_prefix(q)
        if query_role:
            role = query_role
            q = clean_q

        role = (role or "").strip()

        if role:
            url, matched = self._match_in_map(q, self.role_sticker_map.get(role, {}), prefix=role)
            if url:
                return url, matched

            url, matched = self._match_in_map(q, self.common_map, prefix="")
            if url:
                return url, matched

            if self.role_strict:
                return None, None

        # 未设置角色，或允许全库兜底时，在 display_meaning 上匹配。
        return self._match_in_map(q, self.sticker_map, prefix="")

    def _guess_meaning(self, text: str, role: str = "") -> str:
        """兜底补发时，根据用户输入 + AI 输出粗略选择 meaning。"""
        t = (text or "").lower()

        rules: list[tuple[tuple[str, ...], list[str]]] = [
            (("想你", "想我", "思念", "好想", "念你"), ["好想你", "要抱抱", "靠近点"]),
            (("爱你", "喜欢你", "亲亲"), ["爱你爱你", "喜欢", "要抱抱", "送你花"]),
            (("抱抱", "摸摸", "贴贴", "撒娇"), ["要抱抱", "再近点", "靠近点"]),
            (("呜呜", "哭", "委屈", "难过", "心疼"), ["呜呜", "啊...（失落）", "你理理我", "哭哭"]),
            (("早上好", "早安"), ["早上好～", "嗨美女（打招呼）"]),
            (("回来", "我回来了", "到家"), ["你终于回来啦！", "我回来啦"]),
            (("晚安", "睡觉", "困了", "睡啦"), ["睡着了", "没睡醒", "886"]),
            (("谢谢", "感谢", "辛苦"), ["谢谢", "你辛苦啦", "送你花"]),
            (("对不起", "抱歉", "错了"), ["对不起", "呜呜"]),
            (("好", "ok", "收到", "可以", "行", "嗯嗯"), ["好的～", "好～", "OK"]),
            (("开心", "高兴", "快乐", "好耶", "耶"), ["开心", "好耶", "嘻嘻", "哈哈可恶"]),
            (("哈哈", "笑死", "乐", "可爱", "嘿嘿", "嘻嘻"), ["哈哈可恶", "嘿嘿", "嘻嘻", "好耶"]),
            (("无语", "离谱", "服了", "烦", "气死"), ["无语", "......", "我晕了"]),
            (("震惊", "啊?", "啊？", "什么", "真的假的", "不会吧", "问号"), ["震惊!!", "啊!?", "真的假的", "一脸懵", "问号"]),
            (("生气", "哼", "不理", "讨厌", "滚"), ["生气", "哼！", "哼！不理你了", "理理我啊"]),
            (("等一下", "等等", "别催", "马上", "处理"), ["别催", "等我处理", "让我想想"]),
            (("吃饭", "饿", "美味", "饭"), ["准备吃饭", "美味"]),
        ]

        for keywords, candidates in rules:
            if any(k in t for k in keywords):
                shuffled = candidates[:]
                random.shuffle(shuffled)
                for candidate in shuffled:
                    if self._match(candidate, role=role)[0]:
                        return candidate

        return ""

    # ═══════════════════════════════════════════════════════════
    # 图片发送缓存：避免 aiocqhttp 发送 URL 图片时被 AstrBot 强制下载失败
    # ═══════════════════════════════════════════════════════════

    def _image_cache_dir(self) -> str:
        path = os.path.join(self._stickers_dir(), ".cache")
        os.makedirs(path, exist_ok=True)
        return path

    def _url_ext(self, url: str) -> str:
        clean = (url or "").split("?", 1)[0].split("#", 1)[0].lower()
        for ext in (".gif", ".png", ".webp", ".jpg", ".jpeg"):
            if clean.endswith(ext):
                return ext
        return ".jpg"

    def _cached_path_for_url(self, url: str) -> str:
        h = hashlib.sha256((url or "").encode("utf-8")).hexdigest()[:24]
        return os.path.join(self._image_cache_dir(), h + self._url_ext(url))

    async def _download_image_to_cache(self, url: str, retries: int = 3) -> str | None:
        """
        先把网络图片下载到本地缓存，再交给 AstrBot 发送本地文件。
        这样可以避开 aiocqhttp 适配器在发送阶段临时下载 URL，导致 ServerDisconnectedError 后整条工具消息失败。
        """
        url = (url or "").strip()
        if not url.startswith(("http://", "https://")):
            return url if os.path.exists(url) else None

        path = self._cached_path_for_url(url)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path

        if aiohttp is None:
            logger.warning("[StickerMaster] aiohttp 不可用，无法下载表情包 URL 到缓存")
            return None

        headers = {
            "User-Agent": "Mozilla/5.0 (StickerMaster/AstrBot)",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Connection": "close",
        }
        timeout = aiohttp.ClientTimeout(total=25, connect=10, sock_read=20)
        last_err: Exception | None = None

        for attempt in range(1, max(1, retries) + 1):
            tmp = path + ".tmp"
            try:
                async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                    async with session.get(url, allow_redirects=True) as resp:
                        if resp.status != 200:
                            raise RuntimeError(f"HTTP {resp.status}")
                        data = await resp.read()
                        if not data:
                            raise RuntimeError("empty image body")
                        with open(tmp, "wb") as f:
                            f.write(data)
                        os.replace(tmp, path)
                        return path
            except Exception as e:
                last_err = e
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
                await asyncio.sleep(0.6 * attempt)

        logger.warning(f"[StickerMaster] 表情包下载失败，url={url}, err={last_err}")
        return None

    async def _image_component_from_url(self, url: str) -> Any | None:
        local_path = await self._download_image_to_cache(url)
        if not local_path:
            return None
        try:
            return Comp.Image.fromFileSystem(local_path)
        except Exception:
            # 兼容极旧版本，没有 fromFileSystem 时退回 fromURL / 本地路径构造。
            try:
                return Comp.Image.fromURL(local_path)
            except Exception:
                return None

    # ═══════════════════════════════════════════════════════════
    # 自定义表情包写入
    # ═══════════════════════════════════════════════════════════

    def _custom_path(self) -> str:
        stickers_dir = self._stickers_dir()
        os.makedirs(stickers_dir, exist_ok=True)
        return os.path.join(stickers_dir, CUSTOM_FILE)

    def _load_custom(self) -> list[dict]:
        p = self._custom_path()
        if not os.path.exists(p):
            return []
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_custom(self, items: list[dict]) -> None:
        p = self._custom_path()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def _strip_command(self, msg: str, *names: str) -> str:
        text = (msg or "").strip()
        for name in names:
            # 同时兼容 /sticker_add、sticker_add、/sticker_import、sticker_import。
            pattern = rf"^\s*/?{re.escape(name)}(?:\s+|$)"
            text = re.sub(pattern, "", text, count=1, flags=re.I).strip()
        return text

    def _parse_sticker_lines(self, body: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
        """
        解析多行粘贴：
          含义 URL
          (角色) 含义 URL
        URL 可以是 jpeg/png/gif/webp 等任意 http(s) 直链。
        返回 ok_list: [(role, meaning, url)] 与 fail_list: [(raw_line, reason)]。
        """
        ok_list: list[tuple[str, str, str]] = []
        fail_list: list[tuple[str, str]] = []

        for raw_line in (body or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue

            match = URL_RE.search(line)
            if not match:
                fail_list.append((line, "没有识别到 http/https URL"))
                continue

            url = match.group(0).strip().rstrip("，,。)")
            meaning_part = line[:match.start()].strip()

            # 允许用户把 URL 放前面：URL 含义
            if not meaning_part:
                meaning_part = line[match.end():].strip()

            if not meaning_part:
                fail_list.append((line, "缺少含义；格式应为：含义 URL"))
                continue

            role, meaning = self._split_role_prefix(meaning_part)
            if not meaning:
                fail_list.append((line, "含义为空"))
                continue

            ok_list.append((role, meaning, url))

        return ok_list, fail_list

    def _handle_add_body(self, body: str) -> str:
        if not body:
            return (
                "用法（单条）：\n"
                "  /sticker_add 含义 直连图片URL\n\n"
                "用法（人格专属）：\n"
                "  /sticker_add (萌猫) 哭哭 https://xxx.gif\n\n"
                "用法（批量，每行一条）：\n"
                "  /sticker_add\n"
                "  (萌猫) 卧槽尼玛理理我啊 https://files.catbox.moe/a2erwr.jpeg\n"
                "  小狗哭哭 https://files.catbox.moe/outz25.gif"
            )

        ok_list, fail_list = self._parse_sticker_lines(body)
        if not ok_list:
            err_lines = "\n".join(f"  ✗ {t}  ←  {r}" for t, r in fail_list)
            return f"❌ 没有可用的条目：\n{err_lines}"

        try:
            items = self._load_custom()
            new_items: list[dict] = []
            seen_update_keys: set[tuple[str, str]] = set()
            incoming = {(role, meaning): url for role, meaning, url in ok_list}

            # 保留未被覆盖的旧条目。
            for item in items:
                role, meaning, url = self._normalise_item(item)
                key = (role, meaning)
                if key in incoming:
                    seen_update_keys.add(key)
                    continue
                if meaning and url:
                    if role:
                        new_items.append({"role": role, "meaning": meaning, "url": url, "category": "自定义"})
                    else:
                        new_items.append({"meaning": meaning, "url": url, "category": "自定义"})

            for role, meaning, url in ok_list:
                if role:
                    new_items.append({"role": role, "meaning": meaning, "url": url, "category": "自定义"})
                else:
                    new_items.append({"meaning": meaning, "url": url, "category": "自定义"})

            self._save_custom(new_items)
            self._reload()
        except Exception as e:
            return f"❌ 保存失败：{e}"

        updated = len(seen_update_keys)
        added = len(ok_list) - updated
        role_count = sum(1 for role, _, _ in ok_list if role)
        lines = [
            f"✅ 完成！新增 {added} 个，更新 {updated} 个，当前共 {len(self.sticker_map)} 个",
            f"  其中本次人格专属：{role_count} 个；通用：{len(ok_list) - role_count} 个",
        ]
        if fail_list:
            lines.append(f"\n⚠️ 以下 {len(fail_list)} 条跳过：")
            for t, r in fail_list:
                lines.append(f"  ✗ {t}  ←  {r}")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    # LLM Tool —— 核心
    # ═══════════════════════════════════════════════════════════

    @_llm_tool(name="send_sticker")
    async def send_sticker(self, event: AstrMessageEvent, meaning: str):
        """
        发送一个真实表情包图片，让对话更自然、更像真人聊天。

        Args:
            meaning(string): 想表达的情绪或场景。优先填表情包库中已有的 meaning。
                如果当前会话设置了角色表情包，也可以传「(人格分组名) 含义」。
        """
        user_text = (getattr(event, "message_str", "") or "").strip()
        if not self._context_allows_sticker(user_text):
            logger.info(f"[StickerMaster] 工具调用被语境门控拦截: '{meaning}'")
            return

        session_key = self._session_key(event)
        role = await self._resolve_current_role(event)

        url, matched = self._match(meaning, role=role)
        if not url:
            logger.info(f"[StickerMaster] 工具调用未匹配到表情包: role='{role}', meaning='{meaning}'，跳过发送")
            return

        local_path = await self._download_image_to_cache(url)
        if not local_path:
            logger.warning(f"[StickerMaster] 工具发送失败：图片源下载失败，matched={matched}, url={url}")
            yield event.plain_result(f"⚠️ 表情包《{matched}》匹配到了，但图片源暂时无法下载，已跳过发送。")
            return

        self._touch_recent(self._tool_sent_event_keys, self._event_key(event))
        self._mark_session_sticker_sent(session_key)
        logger.info(f"[StickerMaster] 工具发送: role={role or '-'}, '{meaning}' → '{matched}' → {url} -> {local_path}")

        yield event.image_result(local_path)
        return

    # ═══════════════════════════════════════════════════════════
    # 工具函数
    # ═══════════════════════════════════════════════════════════

    def _event_key(self, event: AstrMessageEvent) -> str:
        """尽量生成同一轮事件稳定 key；拿不到 message_id 时退回对象 id。"""
        umo = getattr(event, "unified_msg_origin", "") or "unknown_session"
        msg_obj = getattr(event, "message_obj", None)
        for name in ("message_id", "event_id", "id"):
            value = getattr(event, name, None)
            if value:
                return f"{umo}:{value}"
            if msg_obj is not None:
                value = getattr(msg_obj, name, None)
                if value:
                    return f"{umo}:{value}"
        return f"{umo}:{id(event)}"

    def _session_key(self, event: AstrMessageEvent) -> str:
        """同一聊天会话的 key，用于控制连续发表情包的频率。"""
        return getattr(event, "unified_msg_origin", "") or "unknown_session"

    def _in_session_cooldown(self, session_key: str) -> bool:
        if self.cooldown_seconds <= 0:
            return False
        ts = self._last_sticker_session_ts.get(session_key)
        return bool(ts and time.time() - ts <= self.cooldown_seconds)

    def _mark_session_sticker_sent(self, session_key: str) -> None:
        self._last_sticker_session_ts[session_key] = time.time()
        self._eligible_no_sticker_turns[session_key] = 0

    def _touch_recent(self, store: dict[str, float], key: str) -> None:
        store[key] = time.time()

    def _is_recent(self, store: dict[str, float], key: str, ttl: float) -> bool:
        ts = store.get(key)
        return bool(ts and time.time() - ts <= ttl)

    def _mark_admin_command_once(self, event: AstrMessageEvent, name: str) -> bool:
        """同一条管理命令只处理一次，避免 command 与 raw listener 双触发。"""
        key = f"{self._event_key(event)}:{name}"
        if self._is_recent(self._admin_command_event_keys, key, 120):
            return False
        self._touch_recent(self._admin_command_event_keys, key)
        return True

    def _cleanup_recent(self) -> None:
        now = time.time()
        for store in (self._recent_llm_event_keys, self._tool_sent_event_keys, self._admin_command_event_keys):
            for key, ts in list(store.items()):
                if now - ts > 300:
                    store.pop(key, None)
        for key, ts in list(self._last_sticker_session_ts.items()):
            if now - ts > max(300, self.cooldown_seconds * 3):
                self._last_sticker_session_ts.pop(key, None)
                self._eligible_no_sticker_turns.pop(key, None)

    def _chain_has_image(self, chain: list) -> bool:
        for item in chain:
            cls_name = item.__class__.__name__.lower()
            if "image" in cls_name:
                return True
        return False

    def _chain_to_text(self, chain: list) -> str:
        parts: list[str] = []
        for item in chain:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
            elif item.__class__.__name__.lower() == "plain":
                parts.append(str(item))
        return "\n".join(parts)

    def _format_items(self, items: list[tuple[str, str, str]], limit: int = 120) -> str:
        lines: list[str] = []
        for i, (role, meaning, url) in enumerate(items[:limit], 1):
            prefix = f"({role}) " if role else ""
            lines.append(f"  {i}. {prefix}{meaning}  →  {url}")
        if len(items) > limit:
            lines.append(f"  ... 还有 {len(items) - limit} 个")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    # 管理命令
    # ═══════════════════════════════════════════════════════════

    @filter.command("sticker_reload")
    async def cmd_reload(self, event: AstrMessageEvent):
        """热重载表情包数据，无需重启 AstrBot。"""
        self._reload()
        yield event.plain_result(
            f"✅ 表情包已重载，共 {len(self.sticker_map)} 个可用；"
            f"角色 {len(self.role_sticker_map)} 个，通用 {len(self.common_map)} 个"
        )

    @filter.command("sticker_add")
    async def cmd_add(self, event: AstrMessageEvent):
        """添加自定义表情包，支持单条或多行批量。"""
        if not self._mark_admin_command_once(event, "add"):
            return
        msg = event.message_str.strip()
        body = self._strip_command(msg, "sticker_add")
        yield event.plain_result(self._handle_add_body(body))

    @filter.command("sticker_import")
    async def cmd_import(self, event: AstrMessageEvent):
        """批量导入 URL。等同 /sticker_add，名字更直观。"""
        if not self._mark_admin_command_once(event, "add"):
            return
        msg = event.message_str.strip()
        body = self._strip_command(msg, "sticker_import")
        yield event.plain_result(self._handle_add_body(body))

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def raw_multiline_sticker_import(self, event: AstrMessageEvent):
        """
        QQ/部分适配器可能把「第一行是命令，后面多行是内容」处理得不稳定。
        这里用原始消息监听兜底：只拦截第一行单独为 /sticker_add 或 /sticker_import 的多行消息。
        """
        msg = (getattr(event, "message_str", "") or "").strip()
        if not msg or "\n" not in msg:
            return

        lines = msg.splitlines()
        first = lines[0].strip()
        m = re.match(r"^/?(sticker_add|sticker_import)\s*$", first, re.I)
        if not m:
            return

        if not self._mark_admin_command_once(event, "add"):
            return

        body = "\n".join(lines[1:]).strip()
        result = self._handle_add_body(body)
        try:
            event.stop_event()
        except Exception:
            pass
        yield event.plain_result(result)

    @filter.command("sticker_remove")
    async def cmd_remove(self, event: AstrMessageEvent):
        """删除自定义表情包（仅可删除通过 /sticker_add 或 /sticker_import 添加的）。"""
        msg = event.message_str.strip()
        raw = self._strip_command(msg, "sticker_remove")
        role, meaning = self._split_role_prefix(raw)

        if not meaning:
            yield event.plain_result("用法：/sticker_remove 含义\n例：/sticker_remove (萌猫) 哭哭")
            return

        items = self._load_custom()
        new_items = []
        removed = False
        for item in items:
            item_role, item_meaning, item_url = self._normalise_item(item)
            if item_role == role and item_meaning == meaning:
                removed = True
                continue
            if item_meaning and item_url:
                if item_role:
                    new_items.append({"role": item_role, "meaning": item_meaning, "url": item_url, "category": "自定义"})
                else:
                    new_items.append({"meaning": item_meaning, "url": item_url, "category": "自定义"})

        if not removed:
            yield event.plain_result(
                f"❌ 在自定义列表里没找到「{raw}」\n"
                "（只能删除通过 /sticker_add 或 /sticker_import 添加到 custom.json 的表情包）"
            )
            return

        try:
            self._save_custom(new_items)
            self._reload()
        except Exception as e:
            yield event.plain_result(f"❌ 保存失败：{e}")
            return

        yield event.plain_result(
            f"✅ 已删除自定义表情包「{raw}」\n"
            f"  当前共 {len(self.sticker_map)} 个可用"
        )

    @filter.command("sticker_list_custom")
    async def cmd_list_custom(self, event: AstrMessageEvent):
        """查看所有通过 /sticker_add 或 /sticker_import 添加的自定义表情包。"""
        items = []
        for item in self._load_custom():
            role, meaning, url = self._normalise_item(item)
            if meaning and url:
                items.append((role, meaning, url))

        if not items:
            loaded_note = ""
            if self.sticker_map:
                loaded_note = (
                    f"\n\n注意：当前总库其实已加载 {len(self.sticker_map)} 个，"
                    "但它们来自 default.json 或其他 JSON，不属于 custom.json。"
                    "想看全库用 /sticker_list_all，想看统计用 /sticker_stats。"
                )
            yield event.plain_result(
                "📭 还没有通过 /sticker_add 或 /sticker_import 添加的自定义表情包\n"
                "批量添加格式：\n"
                "  /sticker_import\n"
                "  (萌猫) 哭哭 https://xxx.gif\n"
                "  小狗问号 https://xxx.gif"
                + loaded_note
            )
            return

        yield event.plain_result(
            f"📦 自定义表情包 custom.json（共 {len(items)} 个）：\n" + self._format_items(items)
        )

    @filter.command("sticker_list_all")
    async def cmd_list_all(self, event: AstrMessageEvent):
        """查看全库表情包；可用 /sticker_list_all 萌猫 只看某角色。"""
        msg = event.message_str.strip()
        role_filter = self._strip_command(msg, "sticker_list_all").strip()
        items: list[tuple[str, str, str]] = []

        if role_filter:
            for meaning, url in self.role_sticker_map.get(role_filter, {}).items():
                items.append((role_filter, meaning, url))
        else:
            for meaning, url in self.common_map.items():
                items.append(("", meaning, url))
            for role, mp in self.role_sticker_map.items():
                for meaning, url in mp.items():
                    items.append((role, meaning, url))

        if not items:
            yield event.plain_result(f"📭 没找到表情包{('：' + role_filter) if role_filter else ''}")
            return

        title = f"📦 表情包全库「{role_filter}」（共 {len(items)} 个）：" if role_filter else f"📦 表情包全库（共 {len(items)} 个）："
        yield event.plain_result(title + "\n" + self._format_items(items))

    @filter.command("sticker_search")
    async def cmd_search(self, event: AstrMessageEvent):
        """搜索表情包含义。用法：/sticker_search 关键词"""
        msg = event.message_str.strip()
        keyword = self._strip_command(msg, "sticker_search").strip()
        if not keyword:
            yield event.plain_result("用法：/sticker_search 关键词\n例：/sticker_search 思念")
            return

        pool = await self._available_meanings(event)
        results = [m for m in pool if keyword in m]
        if not results:
            results = difflib.get_close_matches(keyword, pool, n=8, cutoff=0.2)

        if results:
            txt = f"🔍 搜索「{keyword}」找到 {len(results)} 个：\n" + "\n".join(
                f"  • {m}" for m in results[:20]
            )
            if len(results) > 20:
                txt += f"\n  ... 还有 {len(results) - 20} 个"
        else:
            txt = f"❌ 未找到含「{keyword}」的表情包"

        yield event.plain_result(txt)

    @filter.command("sticker_roles")
    async def cmd_roles(self, event: AstrMessageEvent):
        """查看可用人格表情包分组。"""
        if not self.role_sticker_map:
            yield event.plain_result("📭 当前还没有人格专属表情包。添加格式：/sticker_import\n(萌猫) 哭哭 https://xxx.gif")
            return

        lines = ["🎭 当前可用人格表情包分组："]
        for role, mp in sorted(self.role_sticker_map.items(), key=lambda x: x[0]):
            lines.append(f"  • {role}：{len(mp)} 个")
        current = await self._resolve_current_role(event)
        lines.append(f"\n当前匹配人格分组：{current or '未匹配'}")
        lines.append("正常不需要手动设置；调试覆盖：/sticker_role 分组名；关闭覆盖：/sticker_role off")
        yield event.plain_result("\n".join(lines))

    @filter.command("sticker_role")
    async def cmd_role(self, event: AstrMessageEvent):
        """设置当前会话表情包角色。"""
        msg = event.message_str.strip()
        role = self._strip_command(msg, "sticker_role").strip()
        session_key = self._session_key(event)

        if not role:
            current = await self._resolve_current_role(event)
            detected = self.session_persona_debug.get(session_key, "未检测")
            yield event.plain_result(
                f"当前匹配人格分组：{current or '未匹配'}\n"
                f"检测到的人格候选：{detected}\n"
                "正常会自动跟随 AstrBot 当前人格；仅调试时用 /sticker_role 分组名 手动覆盖；关闭覆盖：/sticker_role off"
            )
            return

        if role.lower() in {"off", "none", "关闭", "取消", "清空"}:
            self.session_roles.pop(session_key, None)
            yield event.plain_result("✅ 已关闭当前会话的角色表情包限定")
            return

        self.session_roles[session_key] = role
        count = len(self.role_sticker_map.get(role, {}))
        if count:
            yield event.plain_result(f"✅ 当前会话已手动覆盖为「{role}」专属表情包，共 {count} 个")
        else:
            yield event.plain_result(
                f"⚠️ 当前会话已手动覆盖为「{role}」，但库里还没有这个角色的表情包。\n"
                f"添加格式：/sticker_import\n({role}) 哭哭 https://xxx.gif"
            )

    @filter.command("sticker_path")
    async def cmd_path(self, event: AstrMessageEvent):
        """查看 custom.json 实际保存路径，排查装错目录/权限问题。"""
        p = self._custom_path()
        exists = os.path.exists(p)
        custom_count = len(self._load_custom())
        panel_ok, panel_fail = self._parse_sticker_text_block(self.config_sticker_text)
        group_sources: list[Any] = []
        if isinstance(self.config_persona_groups, list):
            group_sources.extend(self.config_persona_groups)
        if isinstance(self.config_role_groups, list):
            group_sources.extend(self.config_role_groups)
        for group in group_sources:
            if not isinstance(group, dict):
                continue
            role_name = str(group.get("persona_name") or group.get("role_name") or group.get("role") or "").strip()
            text = str(group.get("stickers_text") or group.get("text") or "")
            ok, fail = self._parse_sticker_text_block(text, default_role=role_name)
            panel_ok.extend(ok)
            panel_fail.extend(fail)
        yield event.plain_result(
            "📁 表情包保存路径\n"
            f"custom.json：{p}\n"
            f"是否存在：{'是' if exists else '否'}\n"
            f"QQ命令自定义条目：{custom_count} 个\n"
            f"配置面板条目：{len(panel_ok)} 个\n"
            f"配置面板未解析：{len(panel_fail)} 条\n"
            f"当前总可用：{len(self.sticker_map)} 个"
        )

    @filter.command("sticker_config_check")
    async def cmd_config_check(self, event: AstrMessageEvent):
        """检查 WebUI 配置面板粘贴栏是否解析成功。"""
        panel_ok, panel_fail = self._parse_sticker_text_block(self.config_sticker_text)
        group_sources: list[Any] = []
        if isinstance(self.config_persona_groups, list):
            group_sources.extend(self.config_persona_groups)
        if isinstance(self.config_role_groups, list):
            group_sources.extend(self.config_role_groups)
        for group in group_sources:
            if not isinstance(group, dict):
                continue
            role_name = str(group.get("persona_name") or group.get("role_name") or group.get("role") or "").strip()
            text = str(group.get("stickers_text") or group.get("text") or "")
            ok, fail = self._parse_sticker_text_block(text, default_role=role_name)
            panel_ok.extend(ok)
            panel_fail.extend(fail)

        lines = [
            f"✅ 配置面板可解析：{len(panel_ok)} 条",
            f"⚠️ 配置面板未解析：{len(panel_fail)} 条",
        ]
        if panel_ok:
            lines.append("\n示例前 10 条：")
            lines.append(self._format_items(panel_ok, limit=10))
        if panel_fail:
            lines.append("\n失败前 10 条：")
            for t, r in panel_fail[:10]:
                lines.append(f"  ✗ {t}  ←  {r}")
        yield event.plain_result("\n".join(lines))

    @filter.command("sticker_persona_check")
    async def cmd_persona_check(self, event: AstrMessageEvent):
        """检查当前 AstrBot 人格是否能匹配到表情包分组。"""
        candidates = await self._detect_persona_candidates(event)
        matched = await self._resolve_current_role(event)
        alias_count = len(self.persona_alias_map)
        lines = [
            "🧩 人格表情包匹配检查",
            f"自动跟随人格：{'开启' if self.auto_persona_enabled else '关闭'}",
            f"AstrBot 人格候选：{', '.join(candidates) if candidates else '未读取到'}",
            f"匹配到的表情包分组：{matched or '未匹配'}",
            f"可用人格分组：{', '.join(sorted(self.role_sticker_map.keys())) if self.role_sticker_map else '暂无'}",
            f"别名数量：{alias_count}",
        ]
        if not matched and self.role_sticker_map:
            lines.append("\n如果没匹配上：在插件配置的人格分组里，把『人格名/人格ID』写成 AstrBot WebUI 里的人格名称，或把实际候选填到『人格别名』里。")
        yield event.plain_result("\n".join(lines))

    @filter.command("sticker_test_url")
    async def cmd_test_url(self, event: AstrMessageEvent):
        """测试某个 URL 或 meaning 是否能下载并发送。"""
        msg = getattr(event, "message_str", "") or ""
        body = self._strip_command(msg, "sticker_test_url").strip()
        if not body:
            yield event.plain_result("用法：/sticker_test_url URL 或 /sticker_test_url 表情含义")
            return

        if body.startswith(("http://", "https://")):
            url = body
            matched = body
        else:
            role = await self._resolve_current_role(event)
            url, matched = self._match(body, role=role)
            if not url:
                yield event.plain_result(f"❌ 没匹配到：{body}")
                return

        local_path = await self._download_image_to_cache(url)
        if not local_path:
            yield event.plain_result(f"❌ 下载失败：{matched}\n{url}")
            return
        yield event.plain_result(f"✅ 下载成功，准备发送：{matched}\n缓存：{local_path}")
        yield event.image_result(local_path)

    @filter.command("sticker_stats")
    async def cmd_stats(self, event: AstrMessageEvent):
        """查看已加载的表情包统计。"""
        custom_count = len(self._load_custom())
        if not self.sticker_map:
            yield event.plain_result("❌ 暂未加载任何表情包，请把 JSON 放入 stickers/ 目录，或用 /sticker_import 添加")
            return

        current_role = await self._resolve_current_role(event)
        role_lines = []
        for role, mp in sorted(self.role_sticker_map.items(), key=lambda x: x[0]):
            role_lines.append(f"  • {role}：{len(mp)} 个")
        if not role_lines:
            role_lines.append("  • 暂无人格专属")

        examples = (await self._available_meanings(event))[:5]
        yield event.plain_result(
            f"📦 表情包统计\n"
            f"  总数：{len(self.sticker_map)} 个\n"
            f"  其中 custom.json：{custom_count} 个\n"
            f"  通用：{len(self.common_map)} 个\n"
            f"  人格分组数：{len(self.role_sticker_map)} 个\n"
            f"  当前匹配人格分组：{current_role or '未设置'}\n"
            f"  人格严格模式：{'开启' if self.role_strict else '关闭'}\n"
            f"  软兜底：{'开启' if self.fallback_enabled else '关闭'}\n"
            f"  兜底概率：{self.sticker_probability}\n"
            f"  兜底冷却：{self.cooldown_seconds} 秒\n"
            f"  候选分数：{self.min_score}\n"
            f"  连续未发轻推：{self.nudge_after_turns} 轮\n"
            f"  强情绪分数：{self.aggressive_score}\n"
            f"\n人格分组明细：\n" + "\n".join(role_lines[:20]) +
            f"\n\n示例（前5个）：\n" + "\n".join(f"  • {m}" for m in examples)
        )
