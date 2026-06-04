"""
sticker_master — AstrBot 表情包 AI 工具插件
============================================
核心理念：LLM Tool Use，不是拦截替换。
AI 读懂上下文，自己决定 "我现在想发哪个表情包"，然后主动调用工具发出去。
结果是两条独立消息：AI 的文字回复 + 单独的表情包图片，真人感 UP。

作者：Anrrow
版本：1.2.0
"""

import json
import os
import difflib
import logging

from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api import llm_tool

logger = logging.getLogger(__name__)

CUSTOM_FILE = "custom.json"


@register(
    "sticker_master",
    "Anrrow",
    "AI 驱动表情包工具 - 让 AI 像真人一样主动发送表情包",
    "1.2.0",
)
class StickerMaster(Star):
    """
    通过 @llm_tool 让 AI 在对话中主动调用 send_sticker()。
    表情包以独立消息发出，文字+图片分离，行为接近真人。

    文件结构：
        sticker_master/
        ├── main.py          ← 本文件
        └── stickers/
            ├── default.json ← 默认表情包数据
            ├── custom.json  ← 用户自定义（/sticker_add 命令写入）
            └── *.json       ← 可以放多个包，自动合并加载
    """

    def __init__(self, context: Context, config: dict):
        super().__init__(context, config)
        self.sticker_map: dict[str, str] = {}   # meaning → url
        self.meanings_list: list[str] = []       # 给 difflib 用
        self._reload()

    # ═══════════════════════════════════════════════════════════
    # 数据加载
    # ═══════════════════════════════════════════════════════════

    def _reload(self) -> None:
        """扫描 stickers/ 目录，加载/热重载所有 .json 表情包文件。"""
        self.sticker_map.clear()

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
                        self.sticker_map[m] = u
                        n += 1
                total += n
                logger.info(f"[StickerMaster] 加载 {fname}: {n} 个")
            except Exception as e:
                logger.error(f"[StickerMaster] 加载 {fname} 失败: {e}")

        self.meanings_list = list(self.sticker_map.keys())
        logger.info(f"[StickerMaster] ✅ 就绪，共 {total} 个表情包")

    # ═══════════════════════════════════════════════════════════
    # 三级匹配算法
    # ═══════════════════════════════════════════════════════════

    def _match(self, query: str) -> tuple[str | None, str | None]:
        """
        精确 → 子串 → difflib 模糊，返回 (url, matched_meaning)。
        任何一级命中就立刻返回，不继续向下。
        """
        if not self.sticker_map:
            return None, None

        q = query.strip()

        # Level 1：精确匹配
        if q in self.sticker_map:
            return self.sticker_map[q], q

        # Level 2：子串（query 包含在 meaning 里，或反过来）
        for m, u in self.sticker_map.items():
            if q in m or m in q:
                return u, m

        # Level 3：difflib 模糊（cutoff 调低到 0.3，宽松匹配中文短语）
        hits = difflib.get_close_matches(q, self.meanings_list, n=1, cutoff=0.3)
        if hits:
            return self.sticker_map[hits[0]], hits[0]

        return None, None

    # ═══════════════════════════════════════════════════════════
    # 自定义表情包写入（供 /sticker_add 命令调用）
    # ═══════════════════════════════════════════════════════════

    def _custom_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "stickers", CUSTOM_FILE)

    def _load_custom(self) -> list:
        p = self._custom_path()
        if not os.path.exists(p):
            return []
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_custom(self, items: list) -> None:
        p = self._custom_path()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    # ═══════════════════════════════════════════════════════════
    # LLM Tool —— 核心
    # ═══════════════════════════════════════════════════════════

    @llm_tool(name="send_sticker")
    async def send_sticker(self, event: AstrMessageEvent, meaning: str):
        """
        发送一个表情包，让对话更自然真实，像真人聊天一样。

        【使用频率要求】
        - 几乎每一条消息都要发，这是基本要求，不是建议
        - 不要等"特别明显"的情绪才发，日常一句话就够了
        - 宁可多发也绝对不要憋着，发表情包是你表达情绪的第一反应
        - 以下任何一种情况都必须发：

        【触发场景（以下任意一条成立就应该发）】
        - 对方说了让你心疼 / 开心 / 无语 / 感动的话
        - 对话里有情绪（撒娇 / 思念 / 生气 / 傲娇 / 开心 / 委屈）
        - 对方打招呼 / 说回来了 / 要睡觉了 / 吃饭了等节点
        - 需要一个轻松活泼的收尾时
        - 对方说了一件有趣的事 / 抱怨了什么 / 分享了心情
        - 沉默之后重新开始聊
        - 日常闲聊中任何感觉合适的时刻

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
            f"✅ 表情包已重载，共 {len(self.sticker_map)} 个可用"
        )

    @filter.command("sticker_add")
    async def cmd_add(self, event: AstrMessageEvent):
        """
        添加自定义表情包，支持单条或多行批量。

        单条：/sticker_add 含义 直连URL
        批量（每行一条，含义和URL之间用空格隔开）：
          /sticker_add
          开心死了 https://i.postimg.cc/xxx/happy.png
          我饿了 https://i.postimg.cc/xxx/hungry.png
          生气了 https://i.postimg.cc/xxx/angry.png

        URL 须为直链图片地址（http/https 开头）。
        含义重复时自动覆盖，添加后立即生效。
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

        # ── 逐行解析 ──────────────────────────────────────────
        ok_list   = []   # [(meaning, url, is_update)]
        fail_list = []   # [(line_text, reason)]

        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line:
                continue  # 跳过空行

            parts = line.rsplit(None, 1)   # 最后一段是 URL，其余是含义
            if len(parts) < 2:
                fail_list.append((line, "格式错误（含义和 URL 之间需要空格）"))
                continue

            meaning, url = parts[0].strip(), parts[1].strip()

            if not meaning:
                fail_list.append((line, "含义为空"))
                continue

            if not (url.startswith("http://") or url.startswith("https://")):
                fail_list.append((line, f"URL 须以 http/https 开头"))
                continue

            ok_list.append((meaning, url))

        if not ok_list:
            err_lines = "\n".join(f"  ✗ {t}  ←  {r}" for t, r in fail_list)
            yield event.plain_result(f"❌ 没有可用的条目：\n{err_lines}")
            return

        # ── 批量写入 custom.json ──────────────────────────────
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

        # ── 汇报结果 ──────────────────────────────────────────
        lines = [f"✅ 完成！新增 {added} 个，更新 {updated} 个，当前共 {len(self.sticker_map)} 个"]

        if fail_list:
            lines.append(f"\n⚠️ 以下 {len(fail_list)} 条跳过：")
            for t, r in fail_list:
                lines.append(f"  ✗ {t}  ←  {r}")

        yield event.plain_result("\n".join(lines))

    @filter.command("sticker_remove")
    async def cmd_remove(self, event: AstrMessageEvent):
        """
        删除自定义表情包（仅可删除通过 /sticker_add 添加的）。
        用法：/sticker_remove 含义
        例：/sticker_remove 开心死了
        """
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

        # 从内存移除（如果 default.json 里有同名的会在下次 reload 时恢复）
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
        """
        搜索表情包含义。用法：/sticker_search 关键词
        例：/sticker_search 生气
        """
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
            f"  示例（前5个）：\n"
            + "\n".join(f"  • {m}" for m in self.meanings_list[:5])
        )