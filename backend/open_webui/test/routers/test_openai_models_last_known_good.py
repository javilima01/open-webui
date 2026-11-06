"""A failed model-list fetch for one connection must not be read as "that connection has no models".

``send_get_request`` turns every timeout / connection error into ``None``; in
unpatched v0.11.0 that silently produced a shorter model list, which
``get_all_models`` then published to the shared Redis hash (HDEL-ing the
missing models for every worker).  A successful response, including a
legitimately empty one, must still be applied as-is.
"""

from __future__ import annotations

import logging

import pytest

from open_webui.test.conftest import LITELLM_URL, VLLM_URL

pytestmark = pytest.mark.asyncio

ALL = ['claude-sonnet', 'gpt-4o', 'qwen3-32b']


@pytest.fixture
def worker(owui):
    return owui.Worker('A', models_store={})


async def test_successful_fetch_lists_every_connection(worker, user):
    assert await worker.fetch_openai_models(user) == ALL


async def test_transient_failure_keeps_that_connections_last_good_models(worker, user, caplog):
    await worker.fetch_openai_models(user)

    worker.upstream[VLLM_URL] = None  # vLLM model list timed out
    with caplog.at_level(logging.WARNING):
        assert await worker.fetch_openai_models(user) == ALL

    assert 'qwen3-32b' in worker.app.state.OPENAI_MODELS
    assert worker.app.state.OPENAI_MODELS['qwen3-32b']['urlIdx'] == 1
    assert any(VLLM_URL in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


async def test_recovered_connection_is_used_again(worker, user):
    await worker.fetch_openai_models(user)
    worker.upstream[VLLM_URL] = None
    await worker.fetch_openai_models(user)

    worker.upstream[VLLM_URL] = ['qwen3-32b', 'qwen3-coder']
    assert await worker.fetch_openai_models(user) == ['claude-sonnet', 'gpt-4o', 'qwen3-32b', 'qwen3-coder']


async def test_successful_empty_response_removes_that_connections_models(worker, user):
    """Legitimate empty state: the upstream answered and has no models."""
    await worker.fetch_openai_models(user)

    worker.upstream[VLLM_URL] = []
    assert await worker.fetch_openai_models(user) == ['claude-sonnet', 'gpt-4o']

    worker.upstream[VLLM_URL] = None  # a later failure must not resurrect the removed models
    assert await worker.fetch_openai_models(user) == ['claude-sonnet', 'gpt-4o']


async def test_successful_smaller_response_is_applied(worker, user):
    await worker.fetch_openai_models(user)

    worker.upstream[LITELLM_URL] = ['gpt-4o']
    assert await worker.fetch_openai_models(user) == ['gpt-4o', 'qwen3-32b']


async def test_failure_before_any_success_adds_nothing(worker, user):
    worker.upstream[VLLM_URL] = None
    assert await worker.fetch_openai_models(user) == ['claude-sonnet', 'gpt-4o']


async def test_disabled_connection_is_removed_not_preserved(worker, user):
    await worker.fetch_openai_models(user)

    worker.api_configs = {'1': {'enable': False}}
    assert await worker.fetch_openai_models(user) == ['claude-sonnet', 'gpt-4o']


async def test_connection_removed_from_config_is_not_kept(worker, user):
    await worker.fetch_openai_models(user)

    worker.urls = [LITELLM_URL]
    assert await worker.fetch_openai_models(user) == ['claude-sonnet', 'gpt-4o']


async def test_per_user_auth_connections_are_not_preserved(worker, user):
    """With session / OAuth auth each user gets their own list; never reuse another user's."""
    worker.api_configs = {'1': {'auth_type': 'session'}}
    await worker.fetch_openai_models(user)

    worker.upstream[VLLM_URL] = None
    assert await worker.fetch_openai_models(user) == ['claude-sonnet', 'gpt-4o']
