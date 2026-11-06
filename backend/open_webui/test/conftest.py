"""Shared fixtures for the v0.11.0 Redis model-cache hotfix tests.

Every simulated OpenWebUI *worker* gets its own ``RedisDict`` (its own client
and its own process-local state) on one shared Redis, which is how a
multi-pod / multi-worker deployment with ``WEBSOCKET_MANAGER=redis`` behaves.

By default the shared Redis is an in-process fakeredis server with Lua
support.  Set ``OWUI_TEST_REDIS_URL=redis://host:6379/15`` to run the same
tests against a real Redis server; only keys under a random test prefix are
touched and they are removed afterwards.

Run from ``backend/``::

    python -m pytest open_webui/test -q
"""

from __future__ import annotations

import os
import tempfile

# open_webui.env / open_webui.config read these at import time.  Point every
# directory the app writes to at a temp dir: config.py empties STATIC_DIR and
# re-copies it from the frontend build, which would delete the tracked files in
# backend/open_webui/static when no frontend build is present.
_TEST_ROOT = tempfile.mkdtemp(prefix='owui-test-')
os.environ.setdefault('DATA_DIR', os.path.join(_TEST_ROOT, 'data'))
os.environ.setdefault('STATIC_DIR', os.path.join(_TEST_ROOT, 'static'))
os.environ.setdefault('FRONTEND_BUILD_DIR', os.path.join(_TEST_ROOT, 'build'))
os.environ.setdefault('WEBUI_SECRET_KEY', 'owui-test-secret')
for _sub in ('data', 'static', 'build'):
    os.makedirs(os.path.join(_TEST_ROOT, _sub), exist_ok=True)

import contextvars  # noqa: E402  (env must be set before anything imports open_webui)
import importlib  # noqa: E402
import uuid  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import redis  # noqa: E402

MUTATING_COMMANDS = frozenset({'hset', 'hsetnx', 'hmset', 'hdel', 'delete', 'set', 'eval', 'evalsha', 'flushdb'})


class FaultInjectingRedis:
    """Wraps a Redis client and makes chosen commands fail like a dropped connection.

    The exception is raised *above* redis-py's own retry layer, i.e. it
    behaves like a connection that stayed broken until retries ran out.
    """

    def __init__(self, inner):
        self._inner = inner
        self.down = False
        self.fail_mutation_number: int | None = None
        self._mutations = 0
        self.calls: list[str] = []

    def fail_nth_mutation(self, n: int) -> None:
        """Fail the n-th state-changing command issued from now on (1-based)."""
        self._mutations = 0
        self.fail_mutation_number = n

    def heal(self) -> None:
        self.down = False
        self.fail_mutation_number = None
        self._mutations = 0

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            if self.down:
                raise redis.exceptions.ConnectionError('Connection closed by server.')
            if name in MUTATING_COMMANDS:
                self._mutations += 1
                if self.fail_mutation_number == self._mutations:
                    raise redis.exceptions.ConnectionError('Connection closed by server.')
            self.calls.append(name)
            return attr(*args, **kwargs)

        return call


class RedisBackend:
    """One shared Redis (fake or real) that several simulated workers connect to."""

    def __init__(self, url: str | None):
        self.url = url
        self.prefix = f'owui-test-{uuid.uuid4().hex[:10]}'
        if url is None:
            import fakeredis

            self._server = fakeredis.FakeServer()
        self.admin = self.client()

    def client(self):
        if self.url:
            return redis.Redis.from_url(self.url, decode_responses=True)
        import fakeredis

        return fakeredis.FakeRedis(server=self._server, decode_responses=True)

    def key(self, name: str) -> str:
        return f'{self.prefix}:{name}'

    def our_keys(self) -> list[str]:
        return list(self.admin.scan_iter(match=f'*{self.prefix}*'))

    def lose_all_data(self) -> None:
        """Simulate eviction / FLUSHDB / a Redis restart without persistence.

        Only this test's keys are removed, so it is safe on a real server.
        """
        keys = self.our_keys()
        if keys:
            self.admin.delete(*keys)


@pytest.fixture
def redis_backend():
    backend = RedisBackend(os.environ.get('OWUI_TEST_REDIS_URL') or None)
    try:
        yield backend
    finally:
        backend.lose_all_data()


@pytest.fixture
def make_redis_dict(redis_backend, monkeypatch):
    """Build a ``RedisDict`` as a separate worker would: own connection, own state."""
    from open_webui.socket import utils as socket_utils

    def factory(name: str = 'models'):
        client = FaultInjectingRedis(redis_backend.client())
        monkeypatch.setattr(socket_utils, 'get_redis_connection', lambda *a, **k: client)
        return socket_utils.RedisDict(redis_backend.key(name), redis_url='redis://unused'), client

    return factory


# ---------------------------------------------------------------------------
# Simulated OpenWebUI worker for end-to-end model-list / chat lookups
# ---------------------------------------------------------------------------

LITELLM_URL = 'http://litellm:4000/v1'
VLLM_URL = 'http://vllm:8000/v1'

PAST_MODEL_LOOKUP = 'PAST_MODEL_LOOKUP'

# get_openai_runtime_config() takes no request, so the worker making the call is
# tracked here.
CURRENT_WORKER: contextvars.ContextVar = contextvars.ContextVar('CURRENT_WORKER')


class PastModelLookup(Exception):
    """Raised by the first call chat_completion makes *after* the model lookup succeeded."""


class Worker:
    """One OpenWebUI process: its own ``app.state`` and its own view of the upstreams.

    ``upstream[url]`` is the list of model ids that connection returns, or
    ``None`` when the fetch fails (``send_get_request`` returns ``None`` on any
    connection error or timeout).
    """

    def __init__(self, name: str, models_store):
        self.name = name
        self.urls = [LITELLM_URL, VLLM_URL]
        self.api_configs: dict = {}
        self.upstream: dict[str, list[str] | None] = {
            LITELLM_URL: ['gpt-4o', 'claude-sonnet'],
            VLLM_URL: ['qwen3-32b'],
        }
        self.app = SimpleNamespace(
            state=SimpleNamespace(
                MODELS=models_store,
                BASE_MODELS=[],
                OPENAI_MODELS={},
                OLLAMA_MODELS={},
                redis=None,
                _test_worker=self,
            )
        )

    def request(self):
        return SimpleNamespace(app=self.app, state=SimpleNamespace(), cookies={}, headers={})

    async def refresh_models(self, user):
        """What ``GET /api/models`` does on this worker (cache TTL expired)."""
        from open_webui.routers import openai as openai_router
        from open_webui.utils import models as utils_models

        CURRENT_WORKER.set(self)
        await openai_router.get_all_models.cache.clear()
        return await utils_models.get_all_models(self.request(), user=user)

    async def fetch_openai_models(self, user) -> list[str]:
        """The OpenAI-router model fetch alone (``routers.openai.get_all_models``)."""
        from open_webui.routers import openai as openai_router

        CURRENT_WORKER.set(self)
        await openai_router.get_all_models.cache.clear()
        result = await openai_router.get_all_models(self.request(), user=user)
        return sorted(model['id'] for model in result['data'])

    async def chat(self, model_id: str, user) -> str:
        """``POST /api/chat/completions`` up to (and including) the model lookup.

        Returns the HTTP error detail: ``'Model not found'`` when the lookup
        failed, ``PAST_MODEL_LOOKUP`` when the model was found.
        """
        from fastapi import HTTPException

        from open_webui.routers import openai as openai_router

        main = importlib.import_module('open_webui.main')  # the package also defines a main() CLI

        CURRENT_WORKER.set(self)
        await openai_router.get_all_models.cache.clear()
        try:
            await main.chat_completion(
                self.request(),
                {'model': model_id, 'messages': [{'role': 'user', 'content': 'hi'}]},
                user=user,
            )
        except HTTPException as e:
            return e.detail
        raise AssertionError('chat_completion returned without reaching the patched model lookup')


DEFAULT_CONFIG = {
    'openai.enable': True,
    'ollama.enable': False,
    'models.base_models_cache': False,
    'evaluation.arena.enable': False,
    'evaluation.arena.models': [],
    'models.default_metadata': {},
}


class FakeConfig:
    values = DEFAULT_CONFIG

    @classmethod
    async def get_many(cls, *keys):
        return {k: cls.values.get(k) for k in keys}

    @classmethod
    async def get(cls, key, default=None):
        return cls.values.get(key, default)


@pytest.fixture
def user():
    return SimpleNamespace(id='user-1', role='admin', name='Test User', email='test@example.com')


@pytest.fixture
def custom_models():
    """Workspace model rows returned by ``Models.get_all_models()`` (empty by default)."""
    return []


@pytest.fixture
def owui(monkeypatch, custom_models):
    """Patch OpenWebUI's DB / config / HTTP edges so the real model-list code runs in-process."""
    importlib.import_module('open_webui.main')  # import once so chat_completion is available
    from open_webui.models.functions import Functions
    from open_webui.models.models import Models
    from open_webui.routers import openai as openai_router
    from open_webui.utils import models as utils_models

    async def fake_runtime_config():
        worker = CURRENT_WORKER.get()
        return True, list(worker.urls), ['sk-test'] * len(worker.urls), worker.api_configs

    async def fake_send_get_request(request=None, url=None, key=None, user=None, config=None):
        model_ids = request.app.state._test_worker.upstream.get(url.removesuffix('/models'))
        if model_ids is None:
            return None  # what send_get_request returns on timeout / connection error
        return {
            'object': 'list',
            'data': [{'id': model_id, 'object': 'model', 'owned_by': 'litellm'} for model_id in model_ids],
        }

    async def fake_get_all_custom_models(*args, **kwargs):
        return list(custom_models)

    async def empty_list(*args, **kwargs):
        return []

    async def empty_dict(*args, **kwargs):
        return {}

    async def past_lookup(*args, **kwargs):
        raise PastModelLookup(PAST_MODEL_LOOKUP)

    monkeypatch.setattr(openai_router, 'get_openai_runtime_config', fake_runtime_config)
    monkeypatch.setattr(openai_router, 'send_get_request', fake_send_get_request)
    monkeypatch.setattr(utils_models, 'Config', FakeConfig)
    monkeypatch.setattr(utils_models, 'get_function_models', empty_list)
    monkeypatch.setattr(Models, 'get_all_models', fake_get_all_custom_models)
    monkeypatch.setattr(Models, 'get_model_by_id', past_lookup)
    monkeypatch.setattr(Functions, 'get_active_function_ids_by_type', empty_list)
    monkeypatch.setattr(Functions, 'get_functions_by_ids', empty_list)
    monkeypatch.setattr(Functions, 'get_function_valves_by_ids', empty_dict)

    return SimpleNamespace(Worker=Worker)
