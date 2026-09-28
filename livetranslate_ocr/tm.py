"""翻译记忆库（Translation Memory）。

直播翻译里重复率极高：问候语、口头禅、"谢谢关注"、"这波太秀了"、游戏术语……
这些走一次 LLM 就够，之后应该从本地直接秒出。命中率上去之后，
整条链路的平均延迟会显著下降，API 花销也跟着降。

存储：SQLite（持久化 + 单文件好备份），启动时把常用的部分预载进内存做模糊匹配。
匹配分两级：
  1. 精确命中（归一化后完全相等）—— 微秒级
  2. 模糊命中（rapidfuzz 相似度 ≥ 阈值）—— 毫秒级，只对够长的句子启用，
     太短的句子模糊匹配极易误命中（"go" 和 "no" 相似度就不低）
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import threading
import time
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from .config import TmCfg, PROJECT_ROOT

log = logging.getLogger(__name__)

# 风格/提示词一旦改动，缓存的译文就不再代表当前策略，需要整体失效。
# 改 translator 里的 prompt 时把这个数字 +1 即可。
PROMPT_VERSION = 1

_PUNCT = re.compile(r"[^\w一-鿿぀-ヿ가-힯]+")
_SPACE = re.compile(r"\s+")

# 句首出现这些词时，译文强依赖上文，缓存复用会翻车。
# 拉丁文字用词边界判断；中文没有词边界，\b 在 CJK 上根本不成立，
# 所以单独用前缀匹配（否则"那个真的太好笑了"会被误判成可独立缓存）。
_DEICTIC_LATIN = re.compile(
    r"^(it|they|them|their|there|he|she|him|his|her|its|here|then|"
    r"so|but|and|because|which|who)\b",
    re.IGNORECASE,
)
# 指示词要分开看：指代对象在句子里的（"this game is broken"）可以独立缓存，
# 指代悬空的（"this is broken"）不行。所以只在后面跟动词/空泛名词时才算依赖上下文。
# 不区分的话，"this game / this guy / this map" 这类高频句全都会被挡在缓存外。
_DEICTIC_DANGLING = re.compile(
    r"^(this|that|these|those|the|these)\s+"
    r"(is|was|were|are|would|will|can|could|should|might|has|had|does|did|"
    r"thing|things|one|ones|guy|dude|stuff|part|way|bit|kind|sort)\b",
    re.IGNORECASE,
)
# 中文这边一律从宽（只要句首是指示词/代词就判定为依赖上下文），
# 不像英文那样区分 "this game"(可缓存) 和 "this is"(不可)。
# 原因：中文没有词边界，不引入分词器就无法可靠区分"这些装备"(有指代对象)
# 和"这些太贵了"(指代悬空)。宁可少缓存几条，也不要复用出错译。
# 影响有限——这条只在源语言是中文时才生效，而常见用法是英译中。
_DEICTIC_CJK = (
    "他们", "她们", "它们", "这个", "那个", "这些", "那些", "这种", "那种",
    "这里", "那里", "这样", "那样", "这件事", "那件事",
    "他", "她", "它", "这", "那", "其", "此",
)
# 句尾挂在这些词上，说明句子被 VAD 从中间切断了
_TRAILING = re.compile(
    r"\b(and|but|or|so|because|that|which|who|when|if|the|a|an|of|to|in|with)$",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    """归一键：大小写、全半角、标点、空白全部抹平，只留下可比对的内容。"""
    t = unicodedata.normalize("NFKC", text).strip().lower()
    t = _PUNCT.sub(" ", t)
    return _SPACE.sub(" ", t).strip()


def is_context_dependent(text: str) -> bool:
    """判断这句话的译文是否离不开上下文。"""
    stripped = text.strip()
    if not stripped:
        return True
    if _DEICTIC_LATIN.match(stripped) or _DEICTIC_DANGLING.match(stripped):
        return True
    if stripped.startswith(_DEICTIC_CJK):
        return True
    if _TRAILING.search(stripped):
        return True
    # 太短的句子通常缺主语，靠上下文补
    if len(normalize(stripped)) < 4:
        return True
    return False


def variant_tag(dst_lang: str, style: str) -> str:
    """缓存分区标识：目标语言 + 风格指纹 + 提示词版本。"""
    digest = hashlib.sha1(style.encode("utf-8")).hexdigest()[:8]
    return f"{dst_lang}|{digest}|v{PROMPT_VERSION}"


@dataclass
class TmHit:
    source: str
    target: str
    score: float          # 100 = 精确命中
    source_kind: str      # llm | seed | user | import


class TranslationMemory:
    def __init__(self, cfg: TmCfg):
        self.cfg = cfg
        self._lock = threading.RLock()      # 保护内存索引
        self._db_lock = threading.Lock()    # 保护 sqlite 连接
        self._rows: dict[tuple[str, str, str], tuple[str, str, str]] = {}
        # scope=(src_lang, variant) -> 归一化后的 key 列表，供 rapidfuzz 用
        self._index: dict[tuple[str, str], list[str]] = {}
        # 命中计数的写缓冲，攒够一批再落盘
        self._pending: list[tuple[str, str, str]] = []
        self._db: sqlite3.Connection | None = None
        if cfg.enabled:
            self._open()

    # ---- 存储 ----
    def _open(self) -> None:
        path = PROJECT_ROOT / self.cfg.db
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS tm (
                src_norm   TEXT NOT NULL,
                src_lang   TEXT NOT NULL,
                variant    TEXT NOT NULL,
                src_raw    TEXT NOT NULL,
                dst        TEXT NOT NULL,
                kind       TEXT NOT NULL DEFAULT 'llm',
                hits       INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                used_at    REAL NOT NULL,
                PRIMARY KEY (src_norm, src_lang, variant)
            )
            """
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS tm_used ON tm (used_at DESC)"
        )
        self._db.commit()
        self._load()

    def _load(self) -> None:
        """把最常用的 N 条预载进内存，模糊匹配只在内存里做。"""
        assert self._db is not None
        cur = self._db.execute(
            "SELECT src_norm, src_lang, variant, src_raw, dst, kind FROM tm "
            "ORDER BY hits DESC, used_at DESC LIMIT ?",
            (self.cfg.max_memory_entries,),
        )
        with self._lock:
            self._rows.clear()
            self._index.clear()
            for src_norm, src_lang, variant, src_raw, dst, kind in cur:
                self._rows[(src_norm, src_lang, variant)] = (src_raw, dst, kind)
                self._index.setdefault((src_lang, variant), []).append(src_norm)
        log.info(
            "翻译记忆库已载入 %d 条（模糊匹配阈值 %d）",
            len(self._rows),
            self.cfg.fuzzy_threshold,
        )

    # ---- 查询 ----
    def lookup(self, text: str, src_lang: str, variant: str) -> TmHit | None:
        if not self.cfg.enabled:
            return None
        norm = normalize(text)
        if not norm:
            return None
        key = (norm, src_lang, variant)

        with self._lock:
            row = self._rows.get(key)
            if row is not None:
                src_raw, dst, kind = row
                self._mark_hit(key)
                return TmHit(src_raw, dst, 100.0, kind)

            # 模糊：只在同语言同风格分区里找，且只对够长的句子开放
            if len(norm) >= 6:
                candidates = self._index.get((src_lang, variant))
                if candidates:
                    match = process.extractOne(
                        norm,
                        candidates,
                        scorer=fuzz.ratio,
                        score_cutoff=self.cfg.fuzzy_threshold,
                    )
                    if match is not None:
                        matched_norm, score, _ = match
                        hit_row = self._rows.get((matched_norm, src_lang, variant))
                        if hit_row is not None:
                            src_raw, dst, kind = hit_row
                            self._mark_hit((matched_norm, src_lang, variant))
                            return TmHit(src_raw, dst, float(score), kind)
        return None

    def _mark_hit(self, key: tuple[str, str, str]) -> None:
        """记一次命中。命中非常频繁，所以先在内存里攒着，够一批再落盘。"""
        if self._db is None:
            return
        with self._lock:
            self._pending.append(key)
            full = len(self._pending) >= 16
        if full:
            self._flush_hits()

    def _flush_hits(self) -> None:
        with self._lock:
            pending, self._pending = self._pending, []
        if self._db is None or not pending:
            return
        now = time.time()
        try:
            with self._db_lock:
                self._db.executemany(
                    "UPDATE tm SET hits = hits + 1, used_at = ? "
                    "WHERE src_norm = ? AND src_lang = ? AND variant = ?",
                    [(now, n, l, v) for (n, l, v) in pending],
                )
                self._db.commit()
        except sqlite3.Error as exc:  # pragma: no cover
            log.debug("刷新命中计数失败：%s", exc)

    # ---- 写入 ----
    def store(
        self,
        text: str,
        src_lang: str,
        variant: str,
        dst: str,
        kind: str = "llm",
        force: bool = False,
    ) -> bool:
        """写入一条记忆。返回是否真的写了（被上下文依赖检查挡下则返回 False）。"""
        if not self.cfg.enabled or not dst:
            return False
        norm = normalize(text)
        if len(norm) < self.cfg.write_min_chars:
            return False
        if not force and not self.cfg.cache_context_dependent and is_context_dependent(text):
            return False

        key = (norm, src_lang, variant)
        now = time.time()
        with self._lock:
            exists = key in self._rows
            self._rows[key] = (text, dst, kind)
            if not exists:
                self._index.setdefault((src_lang, variant), []).append(norm)
        if self._db is None:
            return True
        try:
            with self._db_lock:
                self._db.execute(
                    "INSERT INTO tm (src_norm, src_lang, variant, src_raw, dst, kind, "
                    "hits, created_at, used_at) VALUES (?,?,?,?,?,?,0,?,?) "
                    "ON CONFLICT(src_norm, src_lang, variant) DO UPDATE SET "
                    "dst = excluded.dst, kind = excluded.kind, used_at = excluded.used_at",
                    (norm, src_lang, variant, text, dst, kind, now, now),
                )
                self._db.commit()
        except sqlite3.Error as exc:  # pragma: no cover
            log.warning("写入记忆库失败：%s", exc)
            return False
        return True

    # ---- 维护 ----
    def stats(self) -> dict:
        with self._lock:
            mem = len(self._rows)
        total = 0
        kinds: dict[str, int] = {}
        if self._db is not None:
            try:
                total = self._db.execute("SELECT COUNT(*) FROM tm").fetchone()[0]
                for kind, n in self._db.execute(
                    "SELECT kind, COUNT(*) FROM tm GROUP BY kind"
                ):
                    kinds[kind] = n
            except sqlite3.Error:
                pass
        return {"in_memory": mem, "total": total, "by_kind": kinds}

    def close(self) -> None:
        self._flush_hits()
        if self._db is not None:
            try:
                self._db.commit()
                self._db.close()
            except sqlite3.Error:
                pass
            self._db = None
