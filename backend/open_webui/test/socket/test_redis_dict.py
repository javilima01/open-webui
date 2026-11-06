"""RedisDict behaviour behind the shared ``open-webui:models`` hash.

Each test models two or more OpenWebUI workers as separate ``RedisDict``
instances on one Redis.  Tests marked "#28777" fail on unpatched v0.11.0,
where ``set()`` skips its write based on a fingerprint kept in process memory.
"""

from __future__ import annotations

import itertools
import os
import sys
import threading

import pytest
import redis

from open_webui.utils.json_codec import JSONCodec

FULL = {
    'gpt-4o': {'id': 'gpt-4o', 'name': 'GPT-4o', 'urlIdx': 0},
    'qwen3-32b': {'id': 'qwen3-32b', 'name': 'Qwen3 32B', 'urlIdx': 1},
}
PARTIAL = {'gpt-4o': FULL['gpt-4o']}  # one worker's vLLM fetch was momentarily short
UPDATED = {
    'gpt-4o': {'id': 'gpt-4o', 'name': 'GPT-4o (renamed)', 'urlIdx': 0},
    'mistral-large': {'id': 'mistral-large', 'name': 'Mistral Large', 'urlIdx': 0},
}


def stored(redis_backend, name='models') -> dict:
    """What every worker sees in Redis right now, read by an independent client."""
    raw = redis_backend.admin.hgetall(redis_backend.key(name))
    return {k: JSONCodec.loads(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Valid responses
# ---------------------------------------------------------------------------


class TestValidResponse:
    def test_set_publishes_mapping(self, make_redis_dict, redis_backend):
        models, _ = make_redis_dict()
        models.set(FULL)

        assert stored(redis_backend) == FULL
        assert 'qwen3-32b' in models
        assert models['qwen3-32b'] == FULL['qwen3-32b']
        assert len(models) == 2

    def test_set_replaces_values_and_drops_removed_models(self, make_redis_dict, redis_backend):
        models, _ = make_redis_dict()
        models.set(FULL)
        models.set(UPDATED)

        assert stored(redis_backend) == UPDATED
        assert 'qwen3-32b' not in models

    def test_identical_refresh_does_not_rewrite_the_hash(self, make_redis_dict):
        """Keeps the write de-duplication that the per-process fingerprint was added for."""
        models, client = make_redis_dict()
        models.set(FULL)
        client.calls.clear()

        models.set(dict(FULL))

        written = [c for c in client.calls if c in ('hset', 'hdel', 'eval', 'evalsha', 'delete')]
        assert written == []


# ---------------------------------------------------------------------------
# Valid empty dataset (legitimate invalidation must keep working)
# ---------------------------------------------------------------------------


class TestValidEmptyDataset:
    def test_empty_mapping_clears_models_for_every_worker(self, make_redis_dict):
        """All models genuinely gone (e.g. every base model deactivated): clear, as upstream does."""
        worker_a, _ = make_redis_dict()
        worker_b, _ = make_redis_dict()
        worker_a.set(FULL)

        worker_b.set({})

        assert len(worker_a) == 0
        assert 'gpt-4o' not in worker_a

    def test_clear_is_shared(self, make_redis_dict):
        worker_a, _ = make_redis_dict()
        worker_b, _ = make_redis_dict()
        worker_a.set(FULL)

        worker_b.clear()

        assert len(worker_a) == 0

    def test_models_republished_after_another_worker_cleared_them(self, make_redis_dict, redis_backend):
        """#28777: B clears (models deactivated, then re-activated); A's next refresh must publish."""
        worker_a, _ = make_redis_dict()
        worker_b, _ = make_redis_dict()
        worker_a.set(FULL)
        worker_b.set({})

        worker_a.set(FULL)  # same list A published before

        assert stored(redis_backend) == FULL


# ---------------------------------------------------------------------------
# Cross-worker convergence: the "Model not found" root cause
# ---------------------------------------------------------------------------


class TestCrossWorkerConvergence:
    def test_partial_list_from_another_worker_is_repaired(self, make_redis_dict, redis_backend):
        """#28777: worker B's momentarily partial list must not stick for everyone."""
        worker_a, _ = make_redis_dict()
        worker_b, _ = make_redis_dict()

        worker_a.set(FULL)
        worker_b.set(PARTIAL)
        assert 'qwen3-32b' not in worker_a  # the failure window

        for _ in range(3):  # A's following refreshes produce its unchanged full list
            worker_a.set(FULL)

        assert 'qwen3-32b' in worker_a
        assert stored(redis_backend) == FULL

    def test_hash_lost_from_redis_is_rebuilt_by_next_refresh(self, make_redis_dict, redis_backend):
        """#28777: eviction / FLUSHDB / non-persistent Redis restart must not need an OWUI restart."""
        worker_a, _ = make_redis_dict()
        worker_a.set(FULL)

        redis_backend.lose_all_data()
        assert len(worker_a) == 0

        worker_a.set(FULL)

        assert stored(redis_backend) == FULL

    def test_fresh_process_always_republishes(self, make_redis_dict, redis_backend):
        """Why restarting OpenWebUI 'fixed' the incident: a new process has no remembered fingerprint."""
        worker_a, _ = make_redis_dict()
        worker_b, _ = make_redis_dict()
        worker_a.set(FULL)
        worker_b.set(PARTIAL)

        restarted_a, _ = make_redis_dict()
        restarted_a.set(FULL)

        assert stored(redis_backend) == FULL

    def test_damage_outside_set_is_repaired_once_the_fingerprint_expires(self, make_redis_dict, redis_backend):
        """A field removed outside set() is repaired once the shared fingerprint expires (<= 300 s)."""
        worker_a, _ = make_redis_dict()
        worker_a.set(FULL)
        redis_backend.admin.hdel(redis_backend.key('models'), 'qwen3-32b')

        fingerprints = [k for k in redis_backend.our_keys() if k != redis_backend.key('models')]
        assert len(fingerprints) == 1
        assert 0 < redis_backend.admin.ttl(fingerprints[0]) <= 300

        redis_backend.admin.delete(fingerprints[0])  # what the TTL does
        worker_a.set(FULL)

        assert stored(redis_backend) == FULL


# ---------------------------------------------------------------------------
# Redis connection problems during synchronization
# ---------------------------------------------------------------------------


class TestRedisFailures:
    def test_failed_write_raises_and_keeps_previous_models(self, make_redis_dict, redis_backend):
        worker_a, client = make_redis_dict()
        worker_a.set(FULL)

        client.fail_nth_mutation(1)
        with pytest.raises(redis.exceptions.ConnectionError):
            worker_a.set(UPDATED)

        assert stored(redis_backend) == FULL

    def test_interruption_mid_sync_never_leaves_a_mixed_hash(self, make_redis_dict, redis_backend):
        """The connection drops after the first write command of a refresh.

        Old code wrote with HSET then HDEL; dropping in between left a mix of
        both lists in Redis.  The hash must hold one complete list.
        """
        worker_a, client = make_redis_dict()
        worker_a.set(FULL)

        client.fail_nth_mutation(2)
        try:
            worker_a.set(UPDATED)
        except redis.exceptions.ConnectionError:
            pass

        assert stored(redis_backend) in (FULL, UPDATED)

    def test_redis_down_is_an_error_not_an_empty_model_list(self, make_redis_dict):
        """A broken connection must never read as 'no models' / 'model absent'."""
        worker_a, client = make_redis_dict()
        worker_a.set(FULL)

        client.down = True
        with pytest.raises(redis.exceptions.ConnectionError):
            'gpt-4o' in worker_a  # noqa: B015
        with pytest.raises(redis.exceptions.ConnectionError):
            len(worker_a)
        with pytest.raises(redis.exceptions.ConnectionError):
            worker_a.set(UPDATED)

    def test_write_after_reconnect_is_applied(self, make_redis_dict, redis_backend):
        worker_a, client = make_redis_dict()
        worker_a.set(FULL)

        client.down = True
        with pytest.raises(redis.exceptions.ConnectionError):
            worker_a.set(UPDATED)

        client.heal()
        worker_a.set(UPDATED)

        assert stored(redis_backend) == UPDATED

    def test_redis_restart_while_worker_was_disconnected(self, make_redis_dict, redis_backend):
        """#28777: Redis restarted empty while A was cut off; A's unchanged list must be republished."""
        worker_a, client = make_redis_dict()
        worker_a.set(FULL)

        client.down = True
        redis_backend.lose_all_data()
        client.heal()

        worker_a.set(FULL)

        assert stored(redis_backend) == FULL


# ---------------------------------------------------------------------------
# Concurrent synchronization
# ---------------------------------------------------------------------------


class _StepGate:
    """Client wrapper that lets the test scheduler release one Redis command at a time."""

    def __init__(self, inner):
        self._inner = inner
        self.go = threading.Semaphore(0)
        self.ran = threading.Semaphore(0)

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            self.go.acquire()
            try:
                return attr(*args, **kwargs)
            finally:
                self.ran.release()

        return call


class _SteppedSet:
    """Runs ``redis_dict.set(mapping)`` in a thread that only advances when stepped."""

    def __init__(self, redis_dict, client, mapping):
        self.gate = _StepGate(client._inner)
        client._inner = self.gate
        self.finished = False
        self.error = None

        def run():
            try:
                redis_dict.set(mapping)
            except Exception as e:  # surfaced by the test
                self.error = e
            finally:
                self.finished = True
                self.gate.ran.release()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        self._client = client

    def step(self):
        """Let the worker execute its next Redis command (or finish)."""
        if self.finished:
            return
        self.gate.go.release()
        self.gate.ran.acquire()

    def finish(self):
        while not self.finished:
            self.step()
        self._thread.join()
        self._client._inner = self.gate._inner  # back to a normal client


def _schedules(steps_per_worker: int):
    """Every interleaving of two workers' steps (which worker runs at each step)."""
    total = 2 * steps_per_worker
    for a_steps in itertools.combinations(range(total), steps_per_worker):
        yield ['A' if i in a_steps else 'B' for i in range(total)]


class TestConcurrentSynchronization:
    """Two workers refresh at the same time, tried in every possible command order.

    Only one Redis command is in flight at any moment, so the result does not
    depend on the thread-safety of the Redis test double, and a reader
    snapshot is taken between every pair of commands.
    """

    def test_every_interleaving_keeps_the_hash_complete_and_converges(self, make_redis_dict, redis_backend):
        valid_during = [FULL, UPDATED, PARTIAL]
        failures = []

        for schedule in _schedules(steps_per_worker=4):
            redis_backend.lose_all_data()
            worker_a, client_a = make_redis_dict()
            worker_b, client_b = make_redis_dict()
            worker_a.set(FULL)
            worker_b.set(FULL)

            runs = {
                'A': _SteppedSet(worker_a, client_a, UPDATED),  # A's list changed
                'B': _SteppedSet(worker_b, client_b, PARTIAL),  # B's fetch came back short
            }
            for who in schedule:
                runs[who].step()
                snapshot = stored(redis_backend)
                if snapshot not in valid_during:
                    failures.append((schedule, 'mixed hash', snapshot))
                    break
            for run in runs.values():
                run.finish()
                assert run.error is None, run.error

            if stored(redis_backend) not in (UPDATED, PARTIAL):
                failures.append((schedule, 'final hash is not one complete list', stored(redis_backend)))
                continue

            # Afterwards each worker's next ordinary refresh (its list unchanged)
            # must make Redis match it, whoever happened to write last.
            worker_a.set(UPDATED)
            if stored(redis_backend) != UPDATED:
                failures.append((schedule, "A's refresh was skipped", stored(redis_backend)))
                continue
            worker_b.set(PARTIAL)
            if stored(redis_backend) != PARTIAL:
                failures.append((schedule, "B's refresh was skipped", stored(redis_backend)))

        failed_schedules = {tuple(schedule) for schedule, _, _ in failures}
        assert failures == [], f'{len(failed_schedules)} of 70 interleavings failed, e.g. {failures[0]}'


@pytest.mark.skipif(
    not os.environ.get('OWUI_TEST_REDIS_URL'),
    reason='needs a real Redis: fakeredis Lua scripting is not thread-safe (an EVAL issued after '
    "concurrent EVALs can run with a previous call's ARGV); set OWUI_TEST_REDIS_URL",
)
class TestConcurrentSynchronizationThreaded:
    """Free-running threads against a real Redis server (stress complement to the test above)."""

    @pytest.fixture(autouse=True)
    def _fast_thread_switching(self):
        previous = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        yield
        sys.setswitchinterval(previous)

    def test_readers_only_ever_see_a_complete_model_list(self, make_redis_dict, redis_backend):
        writers = [make_redis_dict()[0] for _ in range(4)]
        writers[0].set(FULL)
        valid = [FULL, UPDATED]
        bad_snapshots = []
        done = threading.Event()

        def write(idx):
            for i in range(150):
                writers[idx].set(valid[(i + idx) % 2])

        def read():
            reader = redis_backend.client()
            name = redis_backend.key('models')
            while not done.is_set():
                raw = reader.hgetall(name)
                snapshot = {k: JSONCodec.loads(v) for k, v in raw.items()}
                if snapshot not in valid:
                    bad_snapshots.append(snapshot)

        readers = [threading.Thread(target=read) for _ in range(2)]
        threads = [threading.Thread(target=write, args=(i,)) for i in range(len(writers))]
        for t in readers + threads:
            t.start()
        for t in threads:
            t.join()
        done.set()
        for t in readers:
            t.join()

        assert bad_snapshots == [], f'{len(bad_snapshots)} mixed/empty snapshots, e.g. {bad_snapshots[:1]}'

    def test_workers_converge_after_concurrent_refreshes(self, make_redis_dict, redis_backend):
        writers = [make_redis_dict()[0] for _ in range(4)]

        def write(idx):
            for i in range(50):
                writers[idx].set(FULL if (i + idx) % 2 else UPDATED)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(len(writers))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for worker in writers:
            worker.set(FULL)
            assert stored(redis_backend) == FULL
            worker.set(UPDATED)
            assert stored(redis_backend) == UPDATED
