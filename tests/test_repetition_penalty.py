"""Repetition / frequency / presence penalties on the exact CPU sampler: the penalty math on candidate logits,
that a much-repeated token is pushed down among the candidates, and that the defaults stay an exact no-op."""

from __future__ import annotations

import numpy as np
import pytest

from tensorfold.engine.exact_sampling import Sampling, choose, choose_rows, penalized
from tensorfold.server.errors import RequestError
from tensorfold.server.request_options import RequestOptions, parse_numbers


def test_defaults_are_a_no_op():
    s = Sampling(seed=1)
    assert not s.has_penalty
    values = np.array([2.0, 1.0, 0.5])
    ids = np.array([10, 20, 30])
    # penalized returns the same array object at the defaults, and passing recent never changes the pick
    assert penalized(values, ids, [10, 10], s) is values
    assert choose(values, ids, 0, s) == choose(values, ids, 0, s, recent=[10, 10, 10])


def test_repetition_penalty_divides_positive_and_multiplies_negative_logits():
    s = Sampling(seed=1, repetition_penalty=2.0)
    values = np.array([4.0, -3.0, 1.0])
    ids = np.array([10, 20, 30])
    out = penalized(values, ids, recent=[10, 20], s=s)
    # token 10 seen, positive -> 4/2 = 2 ; token 20 seen, negative -> -3*2 = -6 ; token 30 unseen -> unchanged
    assert list(out) == [2.0, -6.0, 1.0]
    # a token never seen is untouched
    assert list(penalized(values, ids, recent=[30], s=s)) == [4.0, -3.0, 0.5]


def test_frequency_penalty_scales_with_count_presence_is_once():
    ids = np.array([10, 20, 30])
    values = np.array([5.0, 5.0, 5.0])
    freq = penalized(values, ids, recent=[10, 10, 10, 20], s=Sampling(seed=1, frequency_penalty=0.5))
    assert list(freq) == [5.0 - 1.5, 5.0 - 0.5, 5.0]          # counts 3, 1, 0
    pres = penalized(values, ids, recent=[10, 10, 10, 20], s=Sampling(seed=1, presence_penalty=2.0))
    assert list(pres) == [3.0, 3.0, 5.0]                      # once each for 10 and 20, nothing for 30


def test_penalty_last_n_limits_the_window():
    ids = np.array([10, 20])
    values = np.array([5.0, 5.0])
    s = Sampling(seed=1, frequency_penalty=1.0, penalty_last_n=2)
    # only the last two tokens ([20, 20]) count, so 10 is untouched and 20 is hit twice
    assert list(penalized(values, ids, recent=[10, 10, 20, 20], s=s)) == [5.0, 3.0]


def test_penalty_flips_the_greedy_pick_when_the_top_token_repeats():
    # low temperature makes the logit gap dominate the Gumbel noise, so the pick is deterministic
    ids = np.array([10, 20])
    values = np.array([10.0, 1.0])
    plain = Sampling(seed=7, temperature=0.01, top_k=0, top_p=1.0)
    assert choose(values, ids, 0, plain) == 10                       # highest logit wins
    penal = Sampling(seed=7, temperature=0.01, top_k=0, top_p=1.0, repetition_penalty=100.0)
    assert choose(values, ids, 0, penal, recent=[10, 10, 10]) == 20  # 10 is crushed, 20 now wins
    assert choose(values, ids, 0, penal, recent=[]) == 10            # empty history: no penalty, 10 again


def test_choose_rows_applies_each_rows_own_history():
    ids = np.array([[10, 20], [10, 20]])
    values = np.array([[10.0, 1.0], [10.0, 1.0]])
    s = Sampling(seed=7, temperature=0.01, top_k=0, top_p=1.0, repetition_penalty=100.0)
    # row 0 has repeated 10 (so it flips to 20); row 1 has no history (stays 10)
    assert choose_rows(values, ids, positions=[0, 1], s=s, recent=[[10, 10], []]) == [20, 10]
    # without recent, both rows keep the greedy pick
    assert choose_rows(values, ids, positions=[0, 1], s=s) == [10, 10]


# -- the request layer: validation and resolving into a Sampling ----------------------------------------------------

@pytest.mark.parametrize("value, words", [(0, "greater than 0"), (-1, "greater than 0"),
                                          ("x", "a finite number"), (float("nan"), "a finite number"),
                                          (True, "a finite number")])
def test_a_bad_repetition_penalty_is_refused(value, words):
    with pytest.raises(RequestError, match=f"repetition_penalty must be {words}"):
        parse_numbers({"repetition_penalty": value})


def test_good_penalty_fields_are_read():
    assert parse_numbers({"repetition_penalty": "1.3"})["repetition_penalty"] == 1.3
    assert parse_numbers({"frequency_penalty": 0.5})["frequency_penalty"] == 0.5
    assert parse_numbers({"presence_penalty": -0.5})["presence_penalty"] == -0.5   # like OpenAI, may be negative
    assert parse_numbers({"penalty_last_n": 64})["penalty_last_n"] == 64
    assert parse_numbers({"penalty_last_n": 2.0})["penalty_last_n"] == 2           # an int-valued float is fine
    assert parse_numbers({"repetition_penalty": None})["repetition_penalty"] is None
    with pytest.raises(RequestError, match="penalty_last_n must be an integer"):
        parse_numbers({"penalty_last_n": 1.5})


def test_the_mac_app_resolves_penalties_into_the_draw():
    options = RequestOptions()
    options.default_sampling = {"temperature": 0.7, "repetition_penalty": 1.1}
    # a server default flows through when the request names none
    assert options._resolve_sampling({"seed": 4}, 0.7, [1, 2]).repetition_penalty == 1.1
    # the request overrides and sets the rest
    s = options._resolve_sampling({"seed": 4, "repetition_penalty": 1.5, "frequency_penalty": 0.2,
                                   "presence_penalty": 0.3, "penalty_last_n": 64}, 0.7, [1, 2])
    assert (s.repetition_penalty, s.frequency_penalty, s.presence_penalty, s.penalty_last_n) == (1.5, 0.2, 0.3, 64)
    assert s.has_penalty
    # no penalty fields: a plain Sampling with the defaults (has_penalty False)
    assert not options._resolve_sampling({"seed": 4, "repetition_penalty": 1.0}, 0.7, [1, 2]).has_penalty


def test_a_penalised_stream_is_opened_with_drafts_off_and_a_plain_one_keeps_them():
    """The decode reroute: a penalty needs each row's history, which the Metal kernel does not carry, so the
    stream the scheduler opens runs serial (no proposer, drafts off) and draws on the CPU exact path.
    Guards the gate where upstream builds the stream (server/prompt_fill.py since 0.6.0): a rebase that
    drops it still passes every other test while drafted rows skip the penalty."""

    from tensorfold.engine.lane_engine import LaneStream
    from tensorfold.server.scheduler import ChatJob, Scheduler
    from tests.lane_fakes import FakeEngine

    def opened(sampling: Sampling | None) -> LaneStream:
        proposers: list[object] = []
        scheduler = Scheduler(FakeEngine(), lanes=2, eos_ids=frozenset({-1}),
                              proposer_factory=lambda: proposers.append(object()) or proposers[-1])
        job = ChatJob("j", [5, 6, 7, 8], 4, 0.7, sampling=sampling)
        scheduler._open_job(job)
        while scheduler._fills:
            scheduler._fill()
        assert job.error is None and job.stream is not None
        return job.stream

    penalised = opened(Sampling(seed=1, temperature=0.7, repetition_penalty=1.3))
    assert penalised.drafts is False and penalised.proposer is None
    for sampling in (None, Sampling(seed=1, temperature=0.7)):
        plain = opened(sampling)
        assert plain.drafts is True and plain.proposer is not None
