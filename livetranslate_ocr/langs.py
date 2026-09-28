"""语言代码 → 人类可读名称，以及维基百科子域映射。"""

from __future__ import annotations

# 语言代码 → 写进 prompt 的名称。名称用目标语言自己写，模型对"用日语写日语"这类
# 指令的服从度比给英文名更高。
LANG_NAMES: dict[str, str] = {
    "zh": "简体中文",
    "zh-tw": "繁体中文",
    "zh-hant": "繁体中文",
    "en": "English",
    "ja": "日本語",
    "ko": "한국어",
    "ru": "Русский",
    "es": "Español",
    "pt": "Português",
    "fr": "Français",
    "de": "Deutsch",
    "it": "Italiano",
    "th": "ไทย",
    "vi": "Tiếng Việt",
    "id": "Bahasa Indonesia",
    "ar": "العربية",
    "tr": "Türkçe",
}

# 语言代码 → 维基百科子域名。没有对应子域的退回英文维基。
WIKI_SUBDOMAIN: dict[str, str] = {
    "zh": "zh",
    "zh-tw": "zh",
    "zh-hant": "zh",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "ru": "ru",
    "es": "es",
    "pt": "pt",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "th": "th",
    "vi": "vi",
    "id": "id",
    "ar": "ar",
    "tr": "tr",
}

# Whisper 用的两位语言代码，与上面的键做归一对齐。
def normalize(code: str) -> str:
    """统一语言代码写法：zh-CN / zh_CN / ZH → zh。"""
    return (code or "").strip().lower().replace("_", "-")


def display_name(code: str) -> str:
    """取语言的可读名称，未知代码原样返回。"""
    c = normalize(code)
    if c in LANG_NAMES:
        return LANG_NAMES[c]
    base = c.split("-", 1)[0]
    return LANG_NAMES.get(base, code)


def wiki_subdomain(code: str, fallback: str = "en") -> str:
    c = normalize(code)
    if c in WIKI_SUBDOMAIN:
        return WIKI_SUBDOMAIN[c]
    return WIKI_SUBDOMAIN.get(c.split("-", 1)[0], fallback)
