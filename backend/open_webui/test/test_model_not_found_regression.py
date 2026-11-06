"""End-to-end regression for the intermittent ``{"detail": "Model not found"}`` incident.

Two OpenWebUI workers share one Redis (``WEBSOCKET_MANAGER=redis``), exactly
like the production pods.  Each test drives the real code path:

* ``Worker.refresh_models`` -> ``utils.models.get_all_models`` (what
  ``GET /api/models`` runs) -> ``routers.openai.get_all_models`` ->
  ``RedisDict.set``;
* ``Worker.chat`` -> ``main.chat_completion`` up to its lookup
  ``if model_id not in request.app.state.MODELS: raise Exception('Model not found')``
  (main.py), which the API returns as HTTP 400 ``{"detail": "Model not found"}``.

Only the outer edges are faked: the HTTP call to LiteLLM / vLLM, the DB rows
and the config store.
"""

from __future__ import annotations

import pytest
import redis

from open_webui.test.conftest import PAST_MODEL_LOOKUP, VLLM_URL

pytestmark = pytest.mark.asyncio

MODEL_NOT_FOUND = 'Model not found'
VLLM_MODEL = 'qwen3-32b'


@pytest.fixture
def cluster(owui, make_redis_dict):
    """Two workers, each with its own RedisDict on the shared Redis (as in production)."""
    models_a, client_a = make_redis_dict()
    models_b, client_b = make_redis_dict()
    worker_a = owui.Worker('A', models_a)
    worker_b = owui.Worker('B', models_b)
    worker_a.redis_client, worker_b.redis_client = client_a, client_b
    return worker_a, worker_b


async def test_incident_sequence_partial_publish_by_one_worker(cluster, user):
    """The suspected incident sequence, with a list that is short *and* looks valid.

    1. Both workers publish the full model list.
    2. Worker B gets a momentarily short list from the gateway (HTTP 200, the
       vLLM model missing) and publishes it; the HDEL removes the model for
       every worker.
    3. Users keep loading the UI on worker A: A refreshes with its unchanged
       full list.  Unpatched v0.11.0 skipped those writes (per-process
       fingerprint), so the model stayed missing on *every* worker until a
       restart.
    """
    worker_a, worker_b = cluster
    await worker_a.refresh_models(user)
    await worker_b.refresh_models(user)
    assert await worker_a.chat(VLLM_MODEL, user) == PAST_MODEL_LOOKUP

    worker_b.upstream[VLLM_URL] = []  # e.g. vLLM / LiteLLM briefly reloading
    await worker_b.refresh_models(user)
    assert await worker_a.chat(VLLM_MODEL, user) == MODEL_NOT_FOUND  # the window we cannot avoid

    worker_b.upstream[VLLM_URL] = [VLLM_MODEL]  # B's gateway is fine again, B is not asked yet
    for _ in range(5):
        await worker_a.refresh_models(user)

    assert await worker_a.chat(VLLM_MODEL, user) == PAST_MODEL_LOOKUP
    assert await worker_b.chat(VLLM_MODEL, user) == PAST_MODEL_LOOKUP


async def test_transient_gateway_failure_on_one_worker_does_not_hide_models(cluster, user):
    """B's vLLM model-list request times out once; no worker may answer "Model not found"."""
    worker_a, worker_b = cluster
    await worker_a.refresh_models(user)
    await worker_b.refresh_models(user)

    worker_b.upstream[VLLM_URL] = None  # send_get_request() -> None (timeout / reset)
    await worker_b.refresh_models(user)

    assert await worker_a.chat(VLLM_MODEL, user) == PAST_MODEL_LOOKUP
    assert await worker_b.chat(VLLM_MODEL, user) == PAST_MODEL_LOOKUP


async def test_models_hash_lost_in_redis_is_rebuilt_without_restart(cluster, user, redis_backend):
    """Redis loses the hash (eviction under memory pressure, FLUSHDB, restart without persistence).

    chat_completion rebuilds an empty MODELS (``if not request.app.state.MODELS``),
    but unpatched v0.11.0 skipped the write because the list matched the
    worker's remembered fingerprint, so every chat failed until a restart.
    """
    worker_a, worker_b = cluster
    await worker_a.refresh_models(user)
    await worker_b.refresh_models(user)

    redis_backend.lose_all_data()

    assert await worker_a.chat(VLLM_MODEL, user) == PAST_MODEL_LOOKUP
    assert await worker_b.chat('gpt-4o', user) == PAST_MODEL_LOOKUP


async def test_redis_write_failure_during_refresh_keeps_models_available(cluster, user):
    """A Redis error while publishing must not cost any worker its models."""
    worker_a, worker_b = cluster
    await worker_a.refresh_models(user)
    await worker_b.refresh_models(user)

    worker_a.upstream[VLLM_URL] = [VLLM_MODEL, 'qwen3-coder']  # a real change, so A must write
    worker_a.redis_client.fail_nth_mutation(1)
    await worker_a.refresh_models(user)  # existing fallback: A keeps an in-process copy

    assert await worker_a.chat(VLLM_MODEL, user) == PAST_MODEL_LOOKUP
    assert await worker_a.chat('qwen3-coder', user) == PAST_MODEL_LOOKUP
    assert await worker_b.chat(VLLM_MODEL, user) == PAST_MODEL_LOOKUP  # shared hash untouched


async def test_sync_recovers_after_redis_reconnects(cluster, user):
    worker_a, worker_b = cluster
    await worker_a.refresh_models(user)
    await worker_b.refresh_models(user)

    worker_b.redis_client.down = True
    worker_b.upstream[VLLM_URL] = [VLLM_MODEL, 'qwen3-coder']
    with pytest.raises(redis.exceptions.ConnectionError):
        await worker_b.chat(VLLM_MODEL, user)  # an outage surfaces as an outage...

    worker_b.redis_client.heal()
    await worker_b.refresh_models(user)

    assert await worker_a.chat('qwen3-coder', user) == PAST_MODEL_LOOKUP  # ...and sync resumes


async def test_redis_read_failure_is_not_reported_as_model_not_found(cluster, user):
    worker_a, _ = cluster
    await worker_a.refresh_models(user)

    worker_a.redis_client.down = True
    with pytest.raises(redis.exceptions.ConnectionError):
        await worker_a.chat(VLLM_MODEL, user)


@pytest.mark.parametrize(
    'custom_models',
    [
        [
            # Admin deactivated every base model (workspace override, same id).
            type('M', (), {'id': mid, 'base_model_id': None, 'is_active': False})()
            for mid in ('gpt-4o', 'claude-sonnet', 'qwen3-32b')
        ]
    ],
)
async def test_deactivating_all_models_still_takes_effect(cluster, user, custom_models):
    """Legitimate empty state: the hotfix must not keep deactivated models selectable."""
    worker_a, worker_b = cluster
    await worker_a.refresh_models(user)  # list is empty after overrides -> MODELS.set({})

    assert await worker_a.chat(VLLM_MODEL, user) == MODEL_NOT_FOUND
    assert await worker_b.chat('gpt-4o', user) == MODEL_NOT_FOUND
