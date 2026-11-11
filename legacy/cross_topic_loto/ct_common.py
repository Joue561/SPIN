# -*- coding: utf-8 -*-
"""
ct_common.py
跨目标（Leave-One-Topic-Out）结构化模式记忆 RAG 立场检测实验 —— 共享工具模块。

包含：
- 命令行通用参数、日志
- UTF-8 JSONL/JSON 读写（坏行容错）、输出覆盖保护
- 标签 / 语义字段 / 候选模式规范化
- 去话题化（pattern-only）替换器
- 分布、JSD、熵
- 评估指标与评估数据加载（07/08 共用）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
LABELS: List[str] = ["favor", "against", "none"]
LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(LABELS)}

LABEL_ALIASES = {
    "favor": "favor", "favour": "favor", "support": "favor", "pro": "favor",
    "支持": "favor", "赞成": "favor", "赞同": "favor",
    "against": "against", "oppose": "against", "con": "against", "反对": "against",
    "none": "none", "neutral": "none", "neither": "none", "中立": "none",
    "无立场": "none", "无关": "none",
}

ALL_METHODS = [
    "current_only", "context_only", "dynamic_semantic", "rag_empty", "context_memory", "source_rag",
    "pattern_only_rag", "transferable_rag", "random_memory_rag", "example_rag", "intopic_rag",
]
# 需要在 04 中单独检索的方法
RAG_METHODS = ["source_rag", "pattern_only_rag", "transferable_rag", "random_memory_rag", "example_rag",
               "intopic_rag"]
# intopic_rag：同话题对照。记忆由“源话题 train + held-out 话题自身 train”构建（不含任何评估样本），
# 检索方式、prompt 格式、评估样本均与 source_rag 完全相同，用于在同一批样本上量化跨话题迁移损失。
# 使用 RAG 指令模板的方法
#   rag_empty      ：RAG 规则与格式完全相同，但记忆为空（分离“指令效应”与“记忆内容效应”）
#   context_memory ：不给语义字段，只给会话上下文 + source_rag 检索到的结构化记忆（检验记忆能否替代显式语义标注）
#   example_rag    ：检索源话题中结构相似的真实样本作为跨话题示例（替代统计分布）
RAG_PROMPT_METHODS = ["rag_empty", "context_memory"] + RAG_METHODS
# 不在 prompt 中给出语义传导字段的方法
NO_SEMANTIC_METHODS = ["current_only", "context_only", "context_memory"]
# 复用其他方法检索结果的方法
RETRIEVAL_SOURCE = {"context_memory": "source_rag"}

# 语义字段
F_REPLY = "llm_reply_relation"
F_PROP = "llm_semantic_propagation"
F_TREL = "llm_target_relation"
F_MAP = "llm_stance_mapping_to_parent"
F_PAT = "llm_candidate_patterns"
F_CONF = "llm_confidence"
SEMANTIC_FIELDS = [F_REPLY, F_PROP, F_TREL, F_MAP, F_PAT]
REQUIRED_SEMANTIC_FIELDS = [F_REPLY, F_PROP, F_TREL]

# 绝不能进入 prompt 的字段（接近标签泄漏）
FORBIDDEN_PROMPT_FIELDS = [
    "stance_label", "stance_label_zh", "llm_inferred_stance_to_main_target",
    "llm_reason_short", "gold_label",
]

WILDCARD = "*"
UNKNOWN = "unknown"

# ---------------------------------------------------------------------------
# 参数 / 日志
# ---------------------------------------------------------------------------

def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {v}")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--overwrite", type=str2bool, default=False,
                        help="是否覆盖已存在的输出文件（默认 false，存在即报错）")
    parser.add_argument("--skip_existing", type=str2bool, default=False,
                        help="输出已完整存在时跳过该步骤/该 topic")
    parser.add_argument("--only_heldout_topic", type=str, nargs="*", default=None,
                        help="只处理指定 held-out topic（原名或目录名均可）")
    parser.add_argument("--debug", type=str2bool, default=False, help="DEBUG 日志")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def setup_logger(name: str, log_dir: Optional[str | Path] = None, debug: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def log_args(logger: logging.Logger, args: argparse.Namespace) -> None:
    logger.info("运行参数: %s", json.dumps(vars(args), ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def read_jsonl(path: str | Path, logger: Optional[logging.Logger] = None) -> List[Dict[str, Any]]:
    """逐行读取 JSONL；坏行记录警告并跳过，不中断流程。"""
    path = Path(path)
    records: List[Dict[str, Any]] = []
    bad = 0
    if not path.exists():
        if logger:
            logger.warning("文件不存在: %s", path)
        return records
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                bad += 1
                if logger and bad <= 20:
                    logger.warning("JSON 解析失败 %s:%d (%s)", path.name, ln, e)
                continue
            if isinstance(obj, dict):
                records.append(obj)
            else:
                bad += 1
    if bad and logger:
        logger.warning("%s 共跳过 %d 条坏行", path, bad)
    return records


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if math.isnan(v) else v
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]], mode: str = "w") -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        f.flush()


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stage_guard(files: Sequence[str | Path], overwrite: bool, skip_existing: bool,
                logger: Optional[logging.Logger] = None, stage: str = "") -> bool:
    """
    输出保护：
      - 均不存在 → True（写）
      - 存在且 overwrite → True
      - 全部存在且 skip_existing → False（跳过）
      - 其他情况 → FileExistsError
    """
    existing = [Path(f) for f in files if Path(f).exists()]
    if not existing:
        return True
    if overwrite:
        if logger:
            logger.info("[%s] --overwrite=true，将覆盖 %d 个已有文件", stage, len(existing))
        return True
    if skip_existing and len(existing) == len(files):
        if logger:
            logger.info("[%s] 输出已存在，--skip_existing=true，跳过", stage)
        return False
    raise FileExistsError(
        f"[{stage}] 以下输出文件已存在: {[str(p) for p in existing[:5]]} ...\n"
        f"如需覆盖请加 --overwrite true；如需跳过已完成的步骤请加 --skip_existing true"
        + ("（注意：当前为部分存在，无法跳过，请检查或覆盖）" if skip_existing else "")
    )


def report_path(path: str | Path, overwrite: bool) -> Path:
    """全局运行报告：已存在且不允许覆盖时，自动改名为 *_YYYYmmdd-HHMMSS.json，避免覆盖旧报告。"""
    import time
    p = Path(path)
    if p.exists() and not overwrite:
        p = p.with_name(f"{p.stem}_{time.strftime('%Y%m%d-%H%M%S')}{p.suffix}")
    return p


def topic_dirname(topic: str) -> str:
    s = re.sub(r"[^\w\-一-鿿]+", "_", str(topic)).strip("_")
    return s or "UNKNOWN_TOPIC"


def list_topic_dirs(base_dir: str | Path, marker_file: str) -> List[Path]:
    base = Path(base_dir)
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir() and (p / marker_file).exists()])


def select_topics(topic_dirs: List[Path], only: Optional[List[str]]) -> List[Path]:
    if not only:
        return topic_dirs
    wanted = {o for o in only} | {topic_dirname(o) for o in only}
    return [d for d in topic_dirs if d.name in wanted]


def heldout_topic_of_dir(topic_dir: Path, splits_dir: Optional[Path] = None) -> str:
    """从 split_report 读取原始 held-out topic 名。"""
    for cand in [topic_dir / "split_report.json",
                 (splits_dir / topic_dir.name / "split_report.json") if splits_dir else None,
                 topic_dir / "memory_build_report.json"]:
        if cand is not None and Path(cand).exists():
            rep = read_json(cand, {})
            if rep.get("heldout_topic"):
                return rep["heldout_topic"]
    return topic_dir.name


# ---------------------------------------------------------------------------
# 规范化
# ---------------------------------------------------------------------------

def normalize_label(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return LABELS[int(x)] if 0 <= int(x) < 3 else None
    s = str(x).strip().lower()
    return LABEL_ALIASES.get(s)


def norm_value(x: Any, default: str = UNKNOWN) -> str:
    """语义字段值规范化：小写、空格→下划线、空值→unknown。"""
    if x is None:
        return default
    if isinstance(x, float) and math.isnan(x):
        return default
    if isinstance(x, (list, tuple)):
        x = "|".join(str(i) for i in x if i is not None)
    elif isinstance(x, dict):
        for k in ("label", "value", "name", "type"):
            if k in x:
                x = x[k]
                break
        else:
            x = json.dumps(x, ensure_ascii=False, sort_keys=True)
    s = str(x).strip().strip('"').strip("'").lower()
    s = re.sub(r"[\s\-]+", "_", s)
    if not s or s in {"nan", "null", "n/a", "na", "unk", "未知"}:
        return default
    return s


def normalize_patterns(x: Any) -> List[str]:
    """候选语言模式规范化为 list：字符串按 | 拆分（兼容 ｜ ; ；和 JSON 字符串）。"""
    if x is None:
        return []
    items: List[Any]
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return normalize_patterns(parsed)
            except json.JSONDecodeError:
                s = s.strip("[]")
        items = re.split(r"[|｜;；]", s)
    elif isinstance(x, (list, tuple)):
        items = []
        for it in x:
            if isinstance(it, dict):
                it = it.get("pattern") or it.get("name") or it.get("pattern_name") or ""
            if isinstance(it, str) and re.search(r"[|｜]", it):
                items.extend(re.split(r"[|｜]", it))
            else:
                items.append(it)
    else:
        items = [x]
    out: List[str] = []
    for it in items:
        v = norm_value(it, default="")
        if v and v not in out and v not in {"none", UNKNOWN}:
            out.append(v)
    return out


def parse_confidence(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        m = re.search(r"\d+(\.\d+)?", str(x))
        if not m:
            return None
        v = float(m.group())
    if math.isnan(v):
        return None
    if v > 1.0 and v <= 100.0:
        v = v / 100.0
    return float(min(max(v, 0.0), 1.0))


def parse_depth(x: Any) -> Optional[int]:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "t", "是"}


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip().lower() in {"", "nan", "null", "none_value"}:
        return True
    if isinstance(v, (list, dict, tuple)) and len(v) == 0:
        return True
    return False


def missing_semantic_fields(rec: Dict[str, Any]) -> List[str]:
    """字段不存在或为空视为缺失（LLM 明确输出 unknown 不算缺失）。"""
    return [f for f in REQUIRED_SEMANTIC_FIELDS if _is_missing(rec.get(f))]


def has_semantic(rec: Dict[str, Any]) -> bool:
    return len(missing_semantic_fields(rec)) == 0


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    if isinstance(x, (list, tuple)):
        return "\n".join(clean_text(i) for i in x if i)
    return str(x).strip()


def short_text(s: str, n: int = 80) -> str:
    s = re.sub(r"\s+", " ", clean_text(s))
    return s if len(s) <= n else s[: n - 1] + "…"


_PUNCT_RE = re.compile(r"[\s　-〿＀-￯!-/:-@\[-`{-~]+")


def text_len(s: str) -> int:
    return len(_PUNCT_RE.sub("", clean_text(s)))


def is_short_text(s: str, max_len: int = 10) -> bool:
    return text_len(s) <= max_len


def char_ngrams(text: str, n: int = 2) -> set:
    t = _PUNCT_RE.sub("", clean_text(text).lower())
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def overlap_ratio(q: set, m: set) -> float:
    if not q or not m:
        return 0.0
    return len(q & m) / len(q)


def stable_id(prefix: str, *parts: Any, length: int = 10) -> str:
    h = hashlib.md5("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{h}"


# ---------------------------------------------------------------------------
# 去话题化
# ---------------------------------------------------------------------------
DEFAULT_TOPIC_LEXICON: Dict[str, List[str]] = {
    "Apollo Go": ["萝卜快跑", "apollo go", "apollo", "阿波罗", "百度", "无人驾驶出租车", "无人驾驶汽车",
                  "无人驾驶", "自动驾驶", "无人车", "萝卜", "robotaxi", "网约车", "出租车", "司机"],
    "Naked Resignation": ["裸辞", "辞职信", "辞职", "离职", "辞工", "跳槽"],
    "Non-marriage Doctrine": ["不婚主义者", "不婚主义", "不婚族", "不婚", "结婚", "婚姻", "彩礼",
                              "生孩子", "生育", "催婚", "单身"],
    "Pre-made Meals": ["预制菜", "预制食品", "预制", "料理包", "中央厨房", "外卖", "现炒"],
    "iPhone 15": ["iphone 15 pro max", "iphone 15 pro", "iphone 15", "iphone15", "iphone", "苹果15",
                  "苹果手机", "苹果公司", "苹果", "果粉", "库克", "华为", "mate60", "mate 60", "安卓",
                  "ios", "手机"],
}
DEFAULT_ATTRIBUTE_TERMS = [
    "性价比", "价格", "价钱", "售价", "安全性", "安全", "质量", "品质", "续航", "电池", "信号", "发热",
    "卫生", "营养", "口感", "味道", "新鲜度", "新鲜", "效率", "体验", "服务", "技术", "工资", "收入",
    "存款", "压力", "自由", "成本", "拍照", "外观", "系统", "芯片", "添加剂", "防腐剂",
]
DEFAULT_CONSEQUENCE_TERMS = [
    "孤独终老", "食品安全问题", "交通事故", "失业", "下岗", "事故", "车祸", "破产", "离婚", "生病",
    "中毒", "涨价", "降价", "倒闭", "焦虑", "抑郁", "后悔", "饿死", "没饭吃", "养老问题", "人口下降",
]


class Detopicalizer:
    """把具体目标词/属性/后果替换为 [TARGET]/[ATTRIBUTE]/[CONSEQUENCE]。"""

    def __init__(self, target_terms: Iterable[str], attribute_terms: Iterable[str],
                 consequence_terms: Iterable[str]):
        def _prep(terms):
            ts = sorted({t.strip().lower() for t in terms if t and t.strip()}, key=len, reverse=True)
            return re.compile("|".join(re.escape(t) for t in ts), re.IGNORECASE) if ts else None

        self.target_terms = sorted({t.strip().lower() for t in target_terms if t and t.strip()}, key=len, reverse=True)
        self._t = _prep(target_terms)
        self._a = _prep(attribute_terms)
        self._c = _prep(consequence_terms)

    @classmethod
    def build(cls, extra_target_terms: Iterable[str] = (), lexicon_path: Optional[str] = None) -> "Detopicalizer":
        lex = {k: list(v) for k, v in DEFAULT_TOPIC_LEXICON.items()}
        attrs = list(DEFAULT_ATTRIBUTE_TERMS)
        cons = list(DEFAULT_CONSEQUENCE_TERMS)
        if lexicon_path and Path(lexicon_path).exists():
            user = read_json(lexicon_path, {})
            for k, v in (user.get("topics") or {}).items():
                lex.setdefault(k, []).extend(v)
            attrs.extend(user.get("attributes", []))
            cons.extend(user.get("consequences", []))
        targets: List[str] = []
        for topic, terms in lex.items():
            targets.append(topic)
            targets.extend(terms)
        targets.extend([t for t in extra_target_terms if t])
        return cls(targets, attrs, cons)

    def __call__(self, text: str) -> str:
        s = clean_text(text)
        if not s:
            return s
        # 占位符保护，防止二次替换
        if self._t:
            s = self._t.sub("\x00T\x00", s)
        if self._c:
            s = self._c.sub("\x00C\x00", s)
        if self._a:
            s = self._a.sub("\x00A\x00", s)
        s = s.replace("\x00T\x00", "[TARGET]").replace("\x00C\x00", "[CONSEQUENCE]").replace("\x00A\x00", "[ATTRIBUTE]")
        s = re.sub(r"(\[TARGET\]){2,}", "[TARGET]", s)
        return s

    def residual_terms(self, text: str) -> List[str]:
        t = clean_text(text).lower()
        return [w for w in self.target_terms if w and w in t]


# ---------------------------------------------------------------------------
# 分布 / 信息论
# ---------------------------------------------------------------------------

def dist_from_counts(counts: Dict[str, float], alpha: float = 0.0) -> List[float]:
    v = np.array([float(counts.get(l, 0.0)) for l in LABELS]) + alpha
    s = v.sum()
    if s <= 0:
        return [1.0 / 3] * 3
    return (v / s).tolist()


def jsd(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence（base 2，取值 [0,1]）。"""
    from scipy.spatial.distance import jensenshannon
    d = jensenshannon(np.asarray(p, float), np.asarray(q, float), base=2)
    if d is None or (isinstance(d, float) and math.isnan(d)):
        return 0.0
    return float(d) ** 2


def norm_entropy(p: Sequence[float]) -> float:
    p = np.asarray(p, float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum() / math.log(len(LABELS)))


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

def compute_metrics(golds: Sequence[str], preds: Sequence[str]) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    n = len(golds)
    if n == 0:
        return {"n": 0}
    p, r, f, s = precision_recall_fscore_support(golds, preds, labels=LABELS, zero_division=0)
    out: Dict[str, Any] = {
        "n": n,
        "accuracy": float(accuracy_score(golds, preds)),
        "macro_f1": float(f1_score(golds, preds, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(golds, preds, labels=LABELS, average="weighted", zero_division=0)),
        "confusion_matrix": {"labels": LABELS,
                             "matrix(rows=gold, cols=pred)": confusion_matrix(golds, preds, labels=LABELS).tolist()},
    }
    for i, l in enumerate(LABELS):
        out[f"{l}_precision"] = float(p[i])
        out[f"{l}_recall"] = float(r[i])
        out[f"{l}_f1"] = float(f[i])
        out[f"{l}_support"] = int(s[i])
    # F_avg：favor 与 against 的 F1 均值（SemEval-2016 等立场检测常用指标，不受 none 类样本极少的影响）
    out["f_avg"] = float((f[0] + f[1]) / 2)
    return out


def complex_subset_flags(rec: Dict[str, Any], short_len: int = 10) -> Dict[str, bool]:
    depth = parse_depth(rec.get("depth"))
    prop = norm_value(rec.get(F_PROP))
    trel = norm_value(rec.get(F_TREL))
    reply = norm_value(rec.get(F_REPLY))
    return {
        "short_text": is_short_text(rec.get("current_text", ""), short_len),
        "is_ambiguous_current": to_bool(rec.get("is_ambiguous_current")),
        "depth_ge_3": depth is not None and depth >= 3,
        "prop_stance_reversal": prop == "stance_reversal",
        "prop_sarcastic_reversal": prop == "sarcastic_reversal",
        "trel_reply_author_only": trel == "reply_author_only",
        "trel_topic_shift": trel == "topic_shift",
        "reply_mock_parent": reply == "mock_parent",
        "reply_disagree_with_parent": reply == "disagree_with_parent",
    }


def load_gold(splits_dir: str | Path, logger: Optional[logging.Logger] = None) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """{(heldout_topic, sample_uid): record} —— 只在评估阶段读取标签。"""
    gold: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for d in list_topic_dirs(splits_dir, "target_eval.jsonl"):
        topic = heldout_topic_of_dir(d)
        for r in read_jsonl(d / "target_eval.jsonl", logger):
            lab = normalize_label(r.get("stance_label"))
            if lab is None:
                continue
            r["_gold"] = lab
            gold[(topic, r["sample_uid"])] = r
    return gold


def load_predictions(predictions_dir: str | Path, logger: Optional[logging.Logger] = None) -> Dict[Tuple[str, str], Dict[str, Dict[str, Any]]]:
    """{(heldout_topic, method): {sample_uid: pred}}；同一样本保留最后一条有效预测。"""
    out: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for f in sorted(Path(predictions_dir).glob("predictions_*.jsonl")):
        for r in read_jsonl(f, logger):
            lab = normalize_label(r.get("predicted_label"))
            if lab is None or not r.get("sample_uid"):
                continue
            r["predicted_label"] = lab
            out.setdefault((r.get("heldout_topic"), r.get("method")), {})[r["sample_uid"]] = r
    return out


def load_retrieval(retrieval_dir: str | Path, logger: Optional[logging.Logger] = None) -> Dict[Tuple[str, str], Dict[str, Dict[str, Any]]]:
    """{(heldout_topic, method): {sample_uid: retrieval_record}}"""
    out: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    base = Path(retrieval_dir)
    if not base.exists():
        return out
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        for f in sorted(d.glob("retrieval_*.jsonl")):
            for r in read_jsonl(f, logger):
                out.setdefault((r.get("heldout_topic"), r.get("method")), {})[r["sample_uid"]] = r
    return out


def safe_mean(xs: Iterable[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else None