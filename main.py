"""
sticker_master — AstrBot 表情包 AI 工具插件
============================================
核心理念：LLM Tool Use + 语境门控 + 软兜底。
AI 拥有第一选择权：适合就主动调用 send_sticker；插件只在“明显适合但 AI 连续没发”的时候轻推一张。

作者：Anrrow
版本：1.4.1-balanced
"""

import json
import os
import difflib
import logging
import random
import time
from typing import Any

from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import ProviderRequest
import astrbot.api.message_components as Comp

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


@register(
    "sticker_master",
    "Anrrow",
    "AI 驱动表情包工具 - 让 AI 像真人一样主动发送表情包",
    "1.4.1-balanced",
)
class StickerMaster(Star):
    """
    通过 LLM Tool 让 AI 主动调用 send_sticker()。
    同时加入“语境门控”：线下模式、严肃讨论、代码排错、论文/合同/长篇分析等场景不发表情包。
    兜底不是硬补，而是“软兜底”：只有在语境明显适合、且 AI 连续几轮没发时才轻推一张。

    文件结构：
        sticker_master/
        ├── main.py
        └── stickers/
            ├── default.json
            ├── custom.json
            └── *.json

    可选配置：
        sticker_probability: 0.0 ~ 1.0，软兜底触发后的补发概率。默认 0.65。
        sticker_sample_size: 每轮给模型看的随机表情包样例数。默认 50。
        sticker_fallback_enabled: 是否启用软兜底。默认 true。
        sticker_cooldown_seconds: 仅用于兜底补发的冷却秒数。默认 45。
        sticker_min_score: 进入“适合发表情包候选轮”的最低情绪分数。默认 2。
        sticker_nudge_after_turns: 连续多少个适合轮 AI 都没发时，兜底才轻推。默认 2。
        sticker_aggressive_score: 情绪分数达到该值时可不等连续轮数，直接进入软兜底。默认 4。
        sticker_debug: 是否输出更详细日志。默认 false。
    """

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context, config or {})
        self.config = config or {}

        self.sticker_map: dict[str, str] = {}   # meaning → url
        self.meanings_list: list[str] = []       # 给 difflib / random 用

        self.sticker_probability = self._cfg_float("sticker_probability", 0.65, 0.0, 1.0)
        self.sticker_sample_size = self._cfg_int("sticker_sample_size", 50, 5, 200)
        self.fallback_enabled = self._cfg_bool("sticker_fallback_enabled", True)
        self.cooldown_seconds = self._cfg_int("sticker_cooldown_seconds", 45, 0, 3600)
        self.min_score = self._cfg_int("sticker_min_score", 2, 1, 10)
        self.nudge_after_turns = self._cfg_int("sticker_nudge_after_turns", 2, 1, 10)
        self.aggressive_score = self._cfg_int("sticker_aggressive_score", 4, 2, 10)
        self.debug = self._cfg_bool("sticker_debug", False)

        # 用于判断“这一轮是不是 LLM 回复”、避免重复补发，以及控制兜底频率。
        self._recent_llm_event_keys: dict[str, float] = {}
        self._tool_sent_event_keys: dict[str, float] = {}
        self._last_sticker_session_ts: dict[str, float] = {}
        self._eligible_no_sticker_turns: dict[str, int] = {}

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

    # ═══════════════════════════════════════════════════════════
    # 数据加载
    # ═══════════════════════════════════════════════════════════

    def _stickers_dir(self) -> str:
        return os.path.join(os.path.dirname(__file__), "stickers")

    def _reload(self) -> None:
        """扫描 stickers/ 目录，加载/热重载所有 .json 表情包文件。"""
        self.sticker_map.clear()

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
                    m = str(item.get("meaning", "")).strip()
                    u = str(item.get("url", "")).strip()
                    if m and u:
                        self.sticker_map[m] = u
                        n += 1
                total += n
                logger.info(f"[StickerMaster] 加载 {fname}: {n} 个")
            except Exception as e:
                logger.error(f"[StickerMaster] 加载 {fname} 失败: {e}")

        self.meanings_list = list(self.sticker_map.keys())
        logger.info(f"[StickerMaster] ✅ 就绪，共 {total} 个表情包")

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

    # ═══════════════════════════════════════════════════════════
    # LLM 请求注入：让模型更愿意调用工具
    # ═══════════════════════════════════════════════════════════

    @filter.on_llm_request()
    async def inject_sticker_hint(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self.meanings_list:
            return

        self._touch_recent(self._recent_llm_event_keys, self._event_key(event))
        self._cleanup_recent()

        sample = random.sample(self.meanings_list, min(self.sticker_sample_size, len(self.meanings_list)))
        hint = (
            "<sticker_master_runtime_hint>\n"
            "你现在拥有 send_sticker 工具，可以发送真实表情包图片。\n"
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

            # 软兜底：不是“适合就立刻补”，而是 AI 连续没发时才轻推。
            # 强情绪场景可以不等连续轮数，防止 AI 又变得完全不用表情包。
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

            meaning = self._guess_meaning(visible_text)
            url, matched = self._match(meaning)
            if not url:
                if self.debug:
                    logger.info(f"[StickerMaster] 兜底跳过：未匹配到 meaning='{meaning}'")
                return

            chain.append(Comp.Image.fromURL(url))
            self._touch_recent(self._tool_sent_event_keys, event_key)
            self._mark_session_sticker_sent(session_key)
            logger.info(f"[StickerMaster] 智能兜底补发: score={score}, '{meaning}' → '{matched}' → {url}")
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
        """
        语境门控：只判断“这一轮适不适合发”。
        注意：这里不是选择具体表情包；具体 meaning 仍由 LLM 或 _guess_meaning 决定。
        """
        user_text = user_text or ""
        reply_text = reply_text or ""
        combined = (user_text + "\n" + reply_text).lower()

        if self._should_skip_by_user_text(user_text):
            return False

        # 长篇说明、代码块、列表教程、结构化输出，发图会打断阅读。
        if len(reply_text) >= 420 or reply_text.count("\n") >= 7:
            return False

        hard_markers = (
            "```", "<code", "</code>", "traceback", "exception",
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
        """
        给“这一轮像不像适合发表情包的聊天瞬间”打分。
        兜底补发必须达到 min_score，避免只靠概率乱补。
        """
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

        # 称呼本身不加分，避免“宝宝，我问个代码问题”也触发。
        # 但短句 + 明显语气词，可以像真人一样偶尔补一个。
        if len(user_text.strip()) <= 25 and len(reply_text.strip()) <= 120:
            if any(x in text for x in ("～", "！", "!!", "？", "??", "呀", "嘛", "啦", "喔")):
                score += 1

        return score

    # ═══════════════════════════════════════════════════════════
    # 三级匹配算法
    # ═══════════════════════════════════════════════════════════

    def _match(self, query: str) -> tuple[str | None, str | None]:
        """
        精确 → 子串 → difflib 模糊，返回 (url, matched_meaning)。
        """
        if not self.sticker_map:
            return None, None

        q = (query or "").strip()
        if not q:
            return None, None

        # Level 1：精确匹配
        if q in self.sticker_map:
            return self.sticker_map[q], q

        # Level 2：子串（query 包含在 meaning 里，或反过来）
        for m, u in self.sticker_map.items():
            if q in m or m in q:
                return u, m

        # Level 3：difflib 模糊
        hits = difflib.get_close_matches(q, self.meanings_list, n=1, cutoff=0.28)
        if hits:
            return self.sticker_map[hits[0]], hits[0]

        return None, None

    def _guess_meaning(self, text: str) -> str:
        """兜底补发时，根据用户输入 + AI 输出粗略选择 meaning。"""
        t = (text or "").lower()

        rules: list[tuple[tuple[str, ...], list[str]]] = [
            (("想你", "想我", "思念", "好想", "念你"), ["好想你", "要抱抱", "靠近点"]),
            (("爱你", "喜欢你", "亲亲"), ["爱你爱你", "喜欢", "要抱抱", "送你花"]),
            (("抱抱", "摸摸", "贴贴", "撒娇"), ["要抱抱", "再近点", "靠近点"]),
            (("呜呜", "哭", "委屈", "难过", "心疼"), ["呜呜", "啊...（失落）", "你理理我"]),
            (("早上好", "早安"), ["早上好～", "嗨美女（打招呼）"]),
            (("回来", "我回来了", "到家"), ["你终于回来啦！", "我回来啦"]),
            (("晚安", "睡觉", "困了", "睡啦"), ["睡着了", "没睡醒", "886"]),
            (("谢谢", "感谢", "辛苦"), ["谢谢", "你辛苦啦", "送你花"]),
            (("对不起", "抱歉", "错了"), ["对不起", "呜呜"]),
            (("好", "ok", "收到", "可以", "行", "嗯嗯"), ["好的～", "好～", "OK"]),
            (("开心", "高兴", "快乐", "好耶", "耶"), ["开心", "好耶", "嘻嘻", "哈哈可恶"]),
            (("哈哈", "笑死", "乐", "可爱", "嘿嘿", "嘻嘻"), ["哈哈可恶", "嘿嘿", "嘻嘻", "好耶"]),
            (("无语", "离谱", "服了", "烦", "气死"), ["无语", "......", "我晕了"]),
            (("震惊", "啊?", "啊？", "什么", "真的假的", "不会吧"), ["震惊!!", "啊!?", "真的假的", "一脸懵"]),
            (("生气", "哼", "不理", "讨厌", "滚"), ["生气", "哼！", "哼！不理你了"]),
            (("等一下", "等等", "别催", "马上", "处理"), ["别催", "等我处理", "让我想想"]),
            (("吃饭", "饿", "美味", "饭"), ["准备吃饭", "美味"]),
        ]

        for keywords, candidates in rules:
            if any(k in t for k in keywords):
                shuffled = candidates[:]
                random.shuffle(shuffled)
                for candidate in shuffled:
                    if self._match(candidate)[0]:
                        return candidate

        return ""

    # ═══════════════════════════════════════════════════════════
    # 自定义表情包写入（供 /sticker_add 命令调用）
    # ═══════════════════════════════════════════════════════════

    def _custom_path(self) -> str:
        stickers_dir = self._stickers_dir()
        os.makedirs(stickers_dir, exist_ok=True)
        return os.path.join(stickers_dir, CUSTOM_FILE)

    def _load_custom(self) -> list:
        p = self._custom_path()
        if not os.path.exists(p):
            return []
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_custom(self, items: list) -> None:
        p = self._custom_path()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    # ═══════════════════════════════════════════════════════════
    # LLM Tool —— 核心
    # ═══════════════════════════════════════════════════════════

    @_llm_tool(name="send_sticker")
    async def send_sticker(self, event: AstrMessageEvent, meaning: str):
        """
        发送一个真实表情包图片，让对话更自然、更像真人聊天。

        使用要求：
        - 先判断聊天场景是否适合，再决定是否调用。
        - 轻松聊天、撒娇、调侃、安慰、打招呼、晚安、庆祝、强情绪表达时可以调用。
        - 线下模式、现实行动模式、严肃讨论、代码排错、论文/合同/数据/长篇分析时不要调用。
        - 不要连续多轮都调用；每次回复最多调用一次。
        - 用户只是称呼“宝宝/老公/老婆”时，不构成调用理由。

        Args:
            meaning(string): 想表达的情绪或场景。优先填表情包库中已有的 meaning，
                也可以自由描述，例如：开心、想你、抱抱、无语、震惊、思考、生气、好的、谢谢、睡觉。
        """
        user_text = (getattr(event, "message_str", "") or "").strip()
        if not self._context_allows_sticker(user_text):
            logger.info(f"[StickerMaster] 工具调用被语境门控拦截: '{meaning}'")
            return

        session_key = self._session_key(event)

        # AI 主动调用代表模型已经做过一次“该不该发”的判断。
        # 这里不再用冷却强拦，避免又回到“AI 老是不发表情包”。
        # 冷却只限制插件兜底，不限制模型主动工具调用。
        url, matched = self._match(meaning)

        if not url:
            # 模型给了很怪的 meaning 时，不再随机乱发，避免语义不贴脸。
            logger.info(f"[StickerMaster] 工具调用未匹配到表情包: '{meaning}'，跳过发送")
            return

        self._touch_recent(self._tool_sent_event_keys, self._event_key(event))
        self._mark_session_sticker_sent(session_key)
        logger.info(f"[StickerMaster] 工具发送: '{meaning}' → '{matched}' → {url}")

        yield event.image_result(url)
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

    def _cleanup_recent(self) -> None:
        now = time.time()
        for store in (self._recent_llm_event_keys, self._tool_sent_event_keys):
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

    # ═══════════════════════════════════════════════════════════
    # 管理命令
    # ═══════════════════════════════════════════════════════════

    @filter.command("sticker_reload")
    async def cmd_reload(self, event: AstrMessageEvent):
        """热重载表情包数据，无需重启 AstrBot。"""
        self._reload()
        yield event.plain_result(
            f"✅ 表情包已重载，共 {len(self.sticker_map)} 个可用"
        )

    @filter.command("sticker_add")
    async def cmd_add(self, event: AstrMessageEvent):
        """
        添加自定义表情包，支持单条或多行批量。

        单条：/sticker_add 含义 直连URL
        批量（每行一条，含义和URL之间用空格隔开）：
          /sticker_add
          开心死了 https://xxx/happy.png
          我饿了 https://xxx/hungry.png
        """
        msg = event.message_str.strip()
        body = msg.replace("/sticker_add", "").replace("sticker_add", "").strip()

        if not body:
            yield event.plain_result(
                "用法（单条）：\n"
                "  /sticker_add 含义 直连图片URL\n\n"
                "用法（批量，每行一条）：\n"
                "  /sticker_add\n"
                "  开心死了 https://xxx.png\n"
                "  我饿了 https://xxx.png\n"
                "  生气了 https://xxx.png"
            )
            return

        ok_list = []
        fail_list = []

        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            parts = line.rsplit(None, 1)
            if len(parts) < 2:
                fail_list.append((line, "格式错误（含义和 URL 之间需要空格）"))
                continue

            meaning, url = parts[0].strip(), parts[1].strip()

            if not meaning:
                fail_list.append((line, "含义为空"))
                continue

            if not (url.startswith("http://") or url.startswith("https://")):
                fail_list.append((line, "URL 须以 http/https 开头"))
                continue

            ok_list.append((meaning, url))

        if not ok_list:
            err_lines = "\n".join(f"  ✗ {t}  ←  {r}" for t, r in fail_list)
            yield event.plain_result(f"❌ 没有可用的条目：\n{err_lines}")
            return

        try:
            items = self._load_custom()
            existing_meanings = {i.get("meaning") for i in items}

            added = updated = 0
            for meaning, url in ok_list:
                is_update = meaning in existing_meanings
                items = [i for i in items if i.get("meaning") != meaning]
                items.append({"meaning": meaning, "url": url, "category": "自定义"})
                self.sticker_map[meaning] = url
                if is_update:
                    updated += 1
                else:
                    added += 1

            self._save_custom(items)
            self.meanings_list = list(self.sticker_map.keys())
        except Exception as e:
            yield event.plain_result(f"❌ 保存失败：{e}")
            return

        lines = [f"✅ 完成！新增 {added} 个，更新 {updated} 个，当前共 {len(self.sticker_map)} 个"]
        if fail_list:
            lines.append(f"\n⚠️ 以下 {len(fail_list)} 条跳过：")
            for t, r in fail_list:
                lines.append(f"  ✗ {t}  ←  {r}")

        yield event.plain_result("\n".join(lines))

    @filter.command("sticker_remove")
    async def cmd_remove(self, event: AstrMessageEvent):
        """删除自定义表情包（仅可删除通过 /sticker_add 添加的）。"""
        msg = event.message_str.strip()
        meaning = msg.replace("/sticker_remove", "").replace("sticker_remove", "").strip()

        if not meaning:
            yield event.plain_result("用法：/sticker_remove 含义\n例：/sticker_remove 开心死了")
            return

        items = self._load_custom()
        new_items = [i for i in items if i.get("meaning") != meaning]

        if len(new_items) == len(items):
            yield event.plain_result(
                f"❌ 在自定义列表里没找到「{meaning}」\n"
                "（只能删除通过 /sticker_add 添加的表情包）"
            )
            return

        try:
            self._save_custom(new_items)
        except Exception as e:
            yield event.plain_result(f"❌ 保存失败：{e}")
            return

        self.sticker_map.pop(meaning, None)
        self.meanings_list = list(self.sticker_map.keys())

        yield event.plain_result(
            f"✅ 已删除自定义表情包「{meaning}」\n"
            f"  当前共 {len(self.sticker_map)} 个可用"
        )

    @filter.command("sticker_list_custom")
    async def cmd_list_custom(self, event: AstrMessageEvent):
        """查看所有通过 /sticker_add 添加的自定义表情包。"""
        items = self._load_custom()
        if not items:
            yield event.plain_result("📭 还没有自定义表情包\n用 /sticker_add 含义 URL 来添加")
            return

        lines = [f"📦 自定义表情包（共 {len(items)} 个）："]
        for i, item in enumerate(items, 1):
            lines.append(f"  {i}. {item.get('meaning', '?')}  →  {item.get('url', '?')}")
        yield event.plain_result("\n".join(lines))

    @filter.command("sticker_search")
    async def cmd_search(self, event: AstrMessageEvent):
        """搜索表情包含义。用法：/sticker_search 关键词"""
        msg = event.message_str.strip()
        keyword = msg.replace("/sticker_search", "").replace("sticker_search", "").strip()
        if not keyword:
            yield event.plain_result("用法：/sticker_search 关键词\n例：/sticker_search 思念")
            return

        results = [m for m in self.meanings_list if keyword in m]
        if not results:
            results = difflib.get_close_matches(keyword, self.meanings_list, n=5, cutoff=0.2)

        if results:
            txt = f"🔍 搜索「{keyword}」找到 {len(results)} 个：\n" + "\n".join(
                f"  • {m}" for m in results[:20]
            )
            if len(results) > 20:
                txt += f"\n  ... 还有 {len(results) - 20} 个"
        else:
            txt = f"❌ 未找到含「{keyword}」的表情包"

        yield event.plain_result(txt)

    @filter.command("sticker_stats")
    async def cmd_stats(self, event: AstrMessageEvent):
        """查看已加载的表情包统计。"""
        custom_count = len(self._load_custom())
        if not self.sticker_map:
            yield event.plain_result("❌ 暂未加载任何表情包，请把 JSON 放入 stickers/ 目录")
            return

        yield event.plain_result(
            f"📦 表情包统计\n"
            f"  总数：{len(self.sticker_map)} 个\n"
            f"  其中自定义：{custom_count} 个\n"
            f"  软兜底：{'开启' if self.fallback_enabled else '关闭'}\n"
            f"  兜底概率：{self.sticker_probability}\n"
            f"  兜底冷却：{self.cooldown_seconds} 秒\n"
            f"  候选分数：{self.min_score}\n"
            f"  连续未发轻推：{self.nudge_after_turns} 轮\n"
            f"  强情绪分数：{self.aggressive_score}\n"
            f"  示例（前5个）：\n"
            + "\n".join(f"  • {m}" for m in self.meanings_list[:5])
        )
