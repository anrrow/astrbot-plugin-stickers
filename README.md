# sticker_master

> AstrBot 插件 · 让 AI 像真人一样主动发送表情包

---

## 核心原理

不是「拦截替换」，是 **LLM Tool Use**。

```
普通方案：  AI 回复文字 → 后处理脚本检测情绪 → 替换/追加表情包
本插件：    AI 读懂上下文 → 主动调用 send_sticker("好想你") → 图片独立发出
```

AI 真正理解「此刻应该发什么」，而不是靠规则匹配。

---

## 安装

1. 把整个 `sticker_master/` 目录放到 AstrBot 的 `plugins/` 目录下
2. 把你的表情包 JSON 文件放进 `stickers/` 子目录，改名为 `default.json`（或任意 `.json` 名）
3. 重启 AstrBot，插件自动注册 `send_sticker` 工具

```
plugins/
└── sticker_master/
    ├── main.py
    ├── _conf_schema.json
    └── stickers/
        └── default.json   ← 你的 stickers_默認.json 改名放这里
```

---

## 表情包 JSON 格式

每个条目需要 `url` 和 `meaning` 两个字段，`category` 可选：

```json
[
  {
    "url": "https://i.postimg.cc/xxx/sticker.png",
    "meaning": "好想你",
    "category": "默認"
  },
  ...
]
```

可以放多个 JSON 文件，插件会自动合并加载。

---

## 匹配算法（三级）

AI 传入 `meaning` 参数后，插件按以下顺序匹配：

| 优先级 | 方式 | 说明 |
|--------|------|------|
| 1 | 精确匹配 | `meaning == "好想你"` |
| 2 | 子串匹配 | `"想你" in "好想你"` 或反向 |
| 3 | difflib 模糊 | 汉明距离，cutoff=0.3 |

任何一级命中即发送，不命中则静默跳过。

---

## 管理命令

| 命令 | 作用 |
|------|------|
| `/sticker_reload` | 热重载所有 JSON，无需重启 |
| `/sticker_search 关键词` | 搜索含指定关键词的表情包含义 |
| `/sticker_stats` | 查看已加载数量和示例 |

---

## 给 AI 的系统提示建议

在你的 AI 系统提示里加一段，效果更自然：

```
你有一个工具叫 send_sticker，可以在聊天中发送表情包图片。
像真人一样自然地使用它——在情绪明显的时刻（撒娇、想念、无语、开心）
偶尔发一个，不要每句都发。发表情包和文字是分开的两条消息。
```

---

## 注意事项

- `Image.fromURL(url)` 需要 AstrBot 所在环境能访问图片 URL
- 如果你用的是 QQ 群/私聊等平台，部分适配器可能需要先下载图片再发
  - 可以把 `await event.send(Image.fromURL(url))` 改为先 `httpx.get(url)` 下载成 bytes 再发
- 表情包太多（400+）不影响性能，加载一次常驻内存，匹配是 O(n) 但 n 很小

---

## 版本历史

- `1.0.0` — 初始版本，三级匹配 + 管理命令
