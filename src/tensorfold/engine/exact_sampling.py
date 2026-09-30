"""Key Gumbel draws by seed, absolute position, and token id so verification matches serial top-k/top-p/min-p
sampling."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Sequence

import numpy as np

MARGIN = 8     # candidates beyond top_k read from the GPU, so tied values resolve by id on the CPU
# top_k 0 (a model that sets only top_p, e.g. Nemotron): candidates read from the GPU for the nucleus
NUCLEUS_CANDIDATES = 256


@dataclass(frozen=True)
class Sampling:
    seed: int
    temperature: float = 1.0
    top_k: int = 20
    top_p: float = 0.95
    min_p: float = 0.0          # keep tokens at least min_p times as likely as the likeliest (after temperature)
    # Penalties on tokens already in the reply, applied to raw logits before top_k/top_p (all no-ops at the defaults):
    repetition_penalty: float = 1.0   # >1 divides a seen token's positive logit / multiplies a negative one (HF rule)
    frequency_penalty: float = 0.0    # subtract this * how many times the token has appeared
    presence_penalty: float = 0.0     # subtract this once if the token has appeared at all
    penalty_last_n: int = 0           # only the last N tokens count (0: the whole reply so far)

    def __post_init__(self) -> None:
        object.__setattr__(self, "top_k", max(0, int(self.top_k)))
        object.__setattr__(self, "penalty_last_n", max(0, int(self.penalty_last_n)))

    @property
    def min_log(self) -> float:
        """ln(min_p), -inf when off: every rule adds it to the row's top scaled logit, one float64 add."""

        return math.log(self.min_p) if self.min_p > 0.0 else -math.inf

    @property
    def has_penalty(self) -> bool:
        """Whether any repetition/frequency/presence penalty is active (else the recent-token history is ignored)."""

        return self.repetition_penalty != 1.0 or self.frequency_penalty != 0.0 or self.presence_penalty != 0.0


def seed_for(tokens: Sequence[int], salt: int = 0) -> int:
    """A reproducible seed from the prompt: the same conversation samples the same reply."""

    digest = hashlib.sha256((",".join(str(int(t)) for t in tokens) + f"|{salt}").encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _mix(x: np.ndarray) -> np.ndarray:
    x = x ^ (x >> np.uint64(30))
    x = x * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27))
    x = x * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def uniform(seed: int, position: int, ids: np.ndarray) -> np.ndarray:
    """Uniform (0, 1) doubles from a splitmix64 hash of (seed, position, token id)."""

    with np.errstate(over="ignore"):
        x = _mix(np.uint64(seed & 0xFFFFFFFFFFFFFFFF) + np.uint64(0x9E3779B97F4A7C15))
        x = _mix(x ^ (np.uint64(position) * np.uint64(0xD1B54A32D192ED03)))
        x = _mix(x ^ ids.astype(np.uint64))
    return (x >> np.uint64(11)).astype(np.float64) * 2.0 ** -53 + 2.0 ** -54


def uniform_rows(seed: int, positions: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """``uniform`` for many positions at once: row r is ``uniform(seed, positions[r], ids[r])``, same bits."""

    with np.errstate(over="ignore"):
        x = _mix(np.uint64(seed & 0xFFFFFFFFFFFFFFFF) + np.uint64(0x9E3779B97F4A7C15))
        x = _mix(x ^ (np.asarray(positions).astype(np.uint64)[:, None] * np.uint64(0xD1B54A32D192ED03)))
        x = _mix(x ^ ids.astype(np.uint64))
    return (x >> np.uint64(11)).astype(np.float64) * 2.0 ** -53 + 2.0 ** -54


def penalized(values: np.ndarray, ids: np.ndarray, recent: Sequence[int] | None, s: Sampling) -> np.ndarray:
    """Candidate ``values`` (the logits for ``ids``) with the repetition / frequency / presence penalties applied for
    the tokens already in the reply (``recent``). Returns ``values`` unchanged when no penalty is active or ``recent``
    is empty. The penalty is applied to the raw logits, before the top_k / top_p / min_p / Gumbel rule, so a
    much-repeated token is pushed down among the candidates."""

    if recent is None or not s.has_penalty:
        return values
    window = recent[-s.penalty_last_n:] if s.penalty_last_n else recent
    if len(window) == 0:
        return values
    counts: dict[int, int] = {}
    for token in window:
        counts[int(token)] = counts.get(int(token), 0) + 1
    seen = np.array([counts.get(int(i), 0) for i in ids], dtype=np.float64)
    out = values.astype(np.float64).copy()
    if s.repetition_penalty != 1.0:
        hit = seen > 0
        out[hit & (out > 0)] /= s.repetition_penalty
        out[hit & (out <= 0)] *= s.repetition_penalty
    if s.frequency_penalty != 0.0 or s.presence_penalty != 0.0:
        out -= s.frequency_penalty * seen + s.presence_penalty * (seen > 0)
    return out


def choose(values: np.ndarray, ids: np.ndarray, position: int, s: Sampling,
           recent: Sequence[int] | None = None) -> int:
    """One row: candidate logits ``values`` for token ``ids`` -> the sampled token id."""

    values = penalized(values, ids, recent, s)
    order = np.lexsort((ids, -values))
    k = max(1, min(int(s.top_k) if s.top_k else len(ids), len(ids)))
    ids = ids[order][:k]
    scaled = values[order][:k].astype(np.float64) / max(float(s.temperature), 1e-6)
    probs = np.exp(scaled - scaled.max())
    probs /= probs.sum()
    if 0.0 < s.top_p < 1.0:
        keep = int(np.searchsorted(np.cumsum(probs), s.top_p) + 1)
        ids, scaled = ids[:keep], scaled[:keep]
    if s.min_p > 0.0:            # a prefix of the order: the tokens within ln(min_p) of the top
        keep = int((scaled >= scaled[0] + s.min_log).sum())
        ids, scaled = ids[:keep], scaled[:keep]
    gumbel = -np.log(-np.log(uniform(s.seed, position, ids)))
    return int(ids[int(np.argmax(scaled + gumbel))])


def choose_rows(values: np.ndarray, ids: np.ndarray, positions: Sequence[int], s: Sampling,
                recent: Sequence[Sequence[int]] | None = None) -> list[int]:
    """Choose each row with the same operation order and bits as an independent ``choose`` call. ``recent`` gives each
    row's reply-so-far tokens for the penalties (row r uses ``recent[r]``); None applies no penalty."""

    rows, width = ids.shape
    if recent is not None and s.has_penalty:
        values = np.stack([penalized(values[r], ids[r], recent[r], s) for r in range(rows)])
    order = np.lexsort((ids, -values), axis=-1)
    k = max(1, min(int(s.top_k) if s.top_k else width, width))
    ids = np.take_along_axis(ids, order, axis=-1)[:, :k]
    scaled = np.take_along_axis(values, order, axis=-1)[:, :k].astype(np.float64) / max(float(s.temperature), 1e-6)
    score = scaled - np.log(-np.log(uniform_rows(s.seed, np.asarray(positions), ids)))
    if 0.0 < s.top_p < 1.0:
        probs = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        keep = (np.cumsum(probs, axis=-1) < s.top_p).sum(axis=-1) + 1    # searchsorted(cumsum, top_p) + 1
        score[np.arange(k)[None, :] >= keep[:, None]] = -np.inf
    if s.min_p > 0.0:
        score[scaled < scaled[:, :1] + s.min_log] = -np.inf               # ``choose``'s min_p prefix
    return [int(t) for t in ids[np.arange(rows), np.argmax(score, axis=-1)]]


def top_candidates(logits: Any, s: Sampling) -> tuple[Any, Any] | None:
    """Return lazy candidates for evaluation with the forward, or None when ``sample_rows`` uses another path."""

    import mlx.core as mx

    from tensorfold.engine.topk import MAX_K, topk_rows

    vocab = int(logits.shape[-1])
    count = min(vocab, (int(s.top_k) if s.top_k else vocab) + MARGIN)
    if count <= MAX_K and logits.dtype == mx.bfloat16:
        return topk_rows(logits.reshape(-1, vocab), count)      # radix select: the exact top by (value, id)
    return None


def sample_rows(logits: Any, positions: Sequence[int], s: Sampling, keep: dict | None = None,
                top: tuple[Any, Any] | None = None, recent: Sequence[Sequence[int]] | None = None) -> list[int]:
    """Sample logits [W, V] at absolute positions, optionally recording candidates in ``keep`` or reusing evaluated
    ``top``. ``recent`` gives each row's reply-so-far tokens for the penalties (None: no penalty)."""

    import mlx.core as mx

    if not s.top_k and 0.0 < s.top_p < 1.0 and top is None and keep is None and not s.has_penalty:
        drawn = _nucleus_rows(logits, positions, s)
        if drawn is not None:
            missing = [r for r, token in enumerate(drawn) if token is None]
            if missing:
                rows = logits.reshape(-1, logits.shape[-1])[mx.array(missing, dtype=mx.int32)]
                fallback = sample_rows(rows, [positions[r] for r in missing], s, keep={})
                for row, token in zip(missing, fallback):
                    drawn[row] = token
            return [int(token) for token in drawn]
    if top is None:
        top = top_candidates(logits, s)
    if top is not None:
        cand, vals = top
    else:
        vocab = int(logits.shape[-1])
        count = min(vocab, (int(s.top_k) if s.top_k else vocab) + MARGIN)
        flat = logits.reshape(-1, vocab).astype(mx.float32)
        if count < vocab:
            cand = mx.argpartition(-flat, kth=count - 1, axis=-1)[:, :count]
        else:
            cand = mx.broadcast_to(mx.arange(vocab)[None, :], flat.shape)
        vals = mx.take_along_axis(flat, cand, axis=-1)
    cand_np, vals_np = np.array(cand), np.array(vals)
    if keep is not None:
        keep["cand"], keep["vals"] = cand_np, vals_np
    return choose_rows(vals_np, cand_np.astype(np.int64), positions, s, recent)


def _nucleus_rows(logits: Any, positions: Sequence[int], s: Sampling) -> list[int | None] | None:
    """Draw each narrow nucleus independently; None marks rows needing the whole vocabulary."""

    import mlx.core as mx

    vocab = int(logits.shape[-1])
    count = min(vocab, NUCLEUS_CANDIDATES)
    if count >= vocab:
        return None
    flat = logits.reshape(-1, vocab).astype(mx.float32)
    temperature = max(float(s.temperature), 1e-6)
    cand = mx.argpartition(-flat, kth=count - 1, axis=-1)[:, :count]
    vals = mx.take_along_axis(flat, cand, axis=-1)
    norm = mx.logsumexp(flat / temperature, axis=-1)
    cand_np, vals_np, norm_np = np.array(cand).astype(np.int64), np.array(vals), np.array(norm).astype(np.float64)
    out: list[int | None] = []
    for row in range(cand_np.shape[0]):
        order = np.lexsort((cand_np[row], -vals_np[row]))
        ids, values = cand_np[row][order], vals_np[row][order].astype(np.float64)
        scaled = values / temperature
        kept = int((np.cumsum(np.exp(scaled - norm_np[row])) < s.top_p).sum()) + 1
        if s.min_p > 0.0:
            kept = min(kept, int((scaled >= scaled[0] + s.min_log).sum()))
        if kept >= count or values[kept - 1] <= values[-1]:
            out.append(None)
            continue
        score = scaled[:kept] - np.log(-np.log(uniform(s.seed, int(positions[row]), ids[:kept])))
        out.append(int(ids[int(np.argmax(score))]))
    return out if any(token is not None for token in out) else None


__all__ = ["MARGIN", "Sampling", "choose", "choose_rows", "penalized", "sample_rows", "seed_for", "top_candidates",
           "uniform", "uniform_rows"]
