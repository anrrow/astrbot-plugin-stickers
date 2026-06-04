"""
sticker_master — AstrBot 表情包 AI 工具插件
============================================
核心理念：LLM Tool Use，不是拦截替换。
AI 读懂上下文，自己决定 "我现在想发哪个表情包"，然后主动调用工具发出去。
结果是两条独立消息：AI 的文字回复 + 单独的表情包图片，真人感 UP。

作者：Anrrow
版本：1.1.0

修复：
- sticker_map / meanings_list 改为私有属性（_sticker_map / _meanings_list）
  防止 AstrBot 框架在初始化时把 config 字段注入覆盖实例属性
  （原因：框架会把 config 里同名字段写入 self，导致 dict 变成 list，
   从而在 .items() 调用时报 'list' object has no attribute 'items'）
- _reload() 改为重新赋值而非 .clear()，彻底保证类型正确
- 新增发送概率控制（STICKER_PROBABILITY，默认 0.85）
"""

import json
import os
import difflib
import logging
import random

from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api import llm_tool

logger = logging.getLogger(__name__)

# ── 发送概率 ──────────────────────────────────────────────────────────────────
# 0.0 = 永不发   1.0 = 每次都发   建议区间：0.75 ~ 1.0
# 改这里就够了，不需要动其他任何地方
STICKER_PROBABILITY: float = 0.95


@register(
    "sticker_master",
    "Anrrow",
    "AI 驱动表情包工具 - 让 AI 像真人一样主动发送表情包",
    "1.1.0",
)
class StickerMaster(Star):
    """
    通过 @llm_tool 让 AI 在对话中主动调用 send_sticker()。
    表情包以独立消息发出，文字+图片分离，行为接近真人。

    文件结构：
        sticker_master/
        ├── main.py          ← 本文件
        └── stickers/
            ├── default.json ← 你的表情包数据（把 stickers_默認.json 改名放进来）
            └── *.json       ← 可以放多个包，自动合并加载
    """

    def __init__(self, context: Context, config: dict):
        super().__init__(context, config)
        # ⚠️ 下划线前缀 = 私有属性，框架不会覆盖
        self._sticker_map: dict[str, str] = {}
        self._meanings_list: list[str] = []
        self._reload()

    # ═══════════════════════════════════════════════════════════
    # 数据加载
    # ═══════════════════════════════════════════════════════════

    def _reload(self) -> None:
        """扫描 stickers/ 目录，加载/热重载所有 .json 表情包文件。"""
        self._sticker_map = {}   # 重新赋值，彻底保证是 dict

        stickers_dir = os.path.join(os.path.dirname(__file__), "stickers")
        if not os.path.exists(stickers_dir):
            os.makedirs(stickers_dir, exist_ok=True)
            logger.warning(
                "[StickerMaster] ⚠️  stickers/ 目录刚创建，"
                "请把你的表情包 JSON 文件放进去再重载"
            )
            return

        total = 0
        for fname in sorted(os.listdir(stickers_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(stickers_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    items = json.load(f)
                n = 0
                for item in items:
                    m = item.get("meaning", "").strip()
                    u = item.get("url", "").strip()
                    if m and u:
                        self._sticker_map[m] = u
                        n += 1
                total += n
                logger.info(f"[StickerMaster] 加载 {fname}: {n} 个")
            except Exception as e:
                logger.error(f"[StickerMaster] 加载 {fname} 失败: {e}")

        self._meanings_list = list(self._sticker_map.keys())
        logger.info(f"[StickerMaster] ✅ 就绪，共 {total} 个表情包")

    # ═══════════════════════════════════════════════════════════
    # 三级匹配算法
    # ═══════════════════════════════════════════════════════════

    def _match(self, query: str) -> tuple[str | None, str | None]:
        """
        精确 → 子串 → difflib 模糊，返回 (url, matched_meaning)。
        任何一级命中就立刻返回，不继续向下。
        """
        if not self._sticker_map:
            return None, None

        q = query.strip()

        # Level 1：精确匹配
        if q in self._sticker_map:
            return self._sticker_map[q], q

        # Level 2：子串（query 包含在 meaning 里，或反过来）
        for m, u in self._sticker_map.items():
            if q in m or m in q:
                return u, m

        # Level 3：difflib 模糊（cutoff 0.3，宽松匹配中文短语）
        hits = difflib.get_close_matches(q, self._meanings_list, n=1, cutoff=0.3)
        if hits:
            return self._sticker_map[hits[0]], hits[0]

        return None, None

    # ═══════════════════════════════════════════════════════════
    # LLM Tool —— 核心
    # ═══════════════════════════════════════════════════════════

    @llm_tool(name="send_sticker")
    async def send_sticker(self, event: AstrMessageEvent, meaning: str):
        """
        发送一个表情包，让对话更自然真实。
        像真人聊天时那样偶尔插图，不要每句都发——克制才有感觉。

        何时使用：
        - 对方说了让你心疼 / 开心 / 无语 / 感动的话
        - 对话里有明显情绪（撒娇 / 想念 / 生气 / 傲娇 / 开心）
        - 需要一个轻松调皮的收尾时
        - 对方打招呼 / 回来了 / 要睡觉了等节点

        Args:
            meaning(string): 想表达的情绪或场景，从以下列表精确选取（效果最好），
                也可以自由描述情绪，系统会模糊匹配最接近的表情包。

                【思念/爱意】
                好想你 / 爱你爱你 / 喜欢 / 我喜欢你 / 好幸福 / 要抱抱 / 拉钩钩 /
                你偷走了我的心 / 送你花 / 俺就亲亲你不做什么 / 给宝贝补补 /
                再近点 / 靠近点

                【撒娇/耍赖】
                呜呜 / 呜呜（撒娇/耍赖）/ 拜托～ / 帮帮我～ / 不要 / 真的不可以吗？/
                最喜欢我吗？/ 哼！不理你了 / 哼！/ 乞讨 / 你理理我

                【等待/焦虑/分离】
                还没好嘛？/ 你在哪里 / 我还想再聊一会儿 / 欸——这就要走了吗？/
                呜呜，你要丢下我吗？/ 啊...（失落）/ 没人理我 / 不回消息试试呢

                【打招呼/节点】
                早上好～ / 我回来啦 / 你终于回来啦！/ 886 / 偷偷观察 / 偷看 /
                嗨美女（打招呼）

                【状态/正在做】
                睡着了 / 没睡醒 / 通勤中 / 上班中 / 学习中 / 喝水中 / 八卦中 /
                看报中 / 悠闲 / 在动脑筋 / 准备吃饭

                【回应/确认】
                好的～ / 好～ / OK / 谢谢 / 对不起 / 你辛苦啦 / ？？/ 你继续说 /
                好羡慕 / 我不会嘛 / 做不了 / relax

                【无语/震惊/懵】
                无语 / ...... / 震惊!! / 一脸懵 / 啊!? / 嗯? / 我晕了 / 哈哈可恶 /
                我操 / 大傻逼 / 疯掉了 / 我操恶俗啊 / 怎会如此 / Damn

                【开心/撒欢】
                嘻嘻 / 嘿嘿 / 好耶 / 爽 / 美味 / 摇尾巴 / 期待 / 感动 /
                好幸福 / 暴富 / 享受 / 一笑了之

                【生气/傲娇/警告】
                生气 / 你敢不听我的？/ 别催 / 等我处理 / 干嘛 / 滚 /
                你踩雷了 / 找人打你 / 找人弄你 / 我不好说话 / 少看扁我 /
                你没资格管 / 禁止发春 / 我拿地球砸死你

                【思考/处理】
                思考 / 让我想想 / 叫什么呢？/ 在动脑筋 / 高雅人士分析

                【搞怪/梗/其他】
                皇帝巡查 / 老公给我买 / 钓你 / 救救我 / 咬你 / 真的假的 /
                886 / 都别活了 / 炸了这个世界 / 放心交给我一定会搞砸的 /
                在干嘛呼吸也要跟我说一声啊 / 那我还活鸡毛啊跳了兄弟
        """
        # ── 概率门控：只改顶部 STICKER_PROBABILITY 常量即可 ──────────────────
        if random.random() > STICKER_PROBABILITY:
            logger.debug(f"[StickerMaster] 概率未命中（{STICKER_PROBABILITY:.0%}），跳过")
            return

        url, matched = self._match(meaning)

        if not url:
            logger.warning(f"[StickerMaster] 无匹配: '{meaning}'，跳过发送")
            return

        logger.info(f"[StickerMaster] '{meaning}' → '{matched}' → {url}")

        yield event.image_result(url)
        return

    # ═══════════════════════════════════════════════════════════
    # 管理命令
    # ═══════════════════════════════════════════════════════════

    @filter.command("sticker_reload")
    async def cmd_reload(self, event: AstrMessageEvent):
        """热重载表情包数据，无需重启 AstrBot。"""
        self._reload()
        yield event.plain_result(
            f"✅ 表情包已重载，共 {len(self._sticker_map)} 个可用"
        )

    @filter.command("sticker_search")
    async def cmd_search(self, event: AstrMessageEvent):
        """
        搜索表情包含义。用法：/sticker_search 关键词
        例：/sticker_search 生气
        """
        msg = event.message_str.strip()
        keyword = msg.replace("/sticker_search", "").replace("sticker_search", "").strip()
        if not keyword:
            yield event.plain_result("用法：/sticker_search 关键词\n例：/sticker_search 思念")
            return

        results = [m for m in self._meanings_list if keyword in m]
        if not results:
            results = difflib.get_close_matches(keyword, self._meanings_list, n=5, cutoff=0.2)

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
        if not self._sticker_map:
            yield event.plain_result("❌ 暂未加载任何表情包，请把 JSON 放入 stickers/ 目录")
            return
        yield event.plain_result(
            f"📦 表情包统计\n"
            f"  总数：{len(self._sticker_map)} 个\n"
            f"  当前发送概率：{STICKER_PROBABILITY:.0%}\n"
            f"  示例（前5个）：\n"
            + "\n".join(f"  • {m}" for m in self._meanings_list[:5])
        )