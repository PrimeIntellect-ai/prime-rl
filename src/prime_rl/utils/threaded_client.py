import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
from openai import AsyncOpenAI
from verifiers.utils.thread_utils import (
    get_or_create_thread_attr,
    get_or_create_thread_loop,
)


class ThreadedAsyncOpenAIClient:
    """
    Wraps AsyncOpenAI, dispatching calls to a ThreadPoolExecutor.
    Each thread maintains its own event loop and client via thread-local storage.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        max_workers: int = 64,
        timeout: int = 1200,
        max_connections: int = 8192,
        max_keepalive_connections: int = 8192,
        max_retries: int = 10,
        headers: dict[str, str] | None = None,
    ):
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="oai-client",
        )
        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout
        self._max_connections = max_connections
        self._max_keepalive_connections = max_keepalive_connections
        self._max_retries = max_retries
        self._headers = headers or {}
        self._tls_key = f"oai_client_{id(self)}"

    @property
    def base_url(self) -> str:
        return self._base_url

    def _create_client(self) -> AsyncOpenAI:
        timeout = httpx.Timeout(self._timeout)
        limits = httpx.Limits(
            max_connections=self._max_connections,
            max_keepalive_connections=self._max_keepalive_connections,
        )
        http_client = httpx.AsyncClient(limits=limits, timeout=timeout, headers=self._headers)
        return AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            max_retries=self._max_retries,
            http_client=http_client,
        )

    def _get_thread_client(self) -> AsyncOpenAI:
        return get_or_create_thread_attr(self._tls_key, self._create_client)

    async def _run_in_thread(self, coro_fn) -> Any:
        def run():
            loop = get_or_create_thread_loop()
            client = self._get_thread_client()
            return loop.run_until_complete(coro_fn(client))

        return await asyncio.get_event_loop().run_in_executor(self.executor, run)

    @property
    def chat(self) -> "_Chat":
        return _Chat(self)

    @property
    def completions(self) -> "_Completions":
        return _Completions(self)

    @property
    def models(self) -> "_Models":
        return _Models(self)

    def teardown(self, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait)


class _Chat:
    def __init__(self, parent: ThreadedAsyncOpenAIClient):
        self._parent = parent

    @property
    def completions(self) -> "_ChatCompletions":
        return _ChatCompletions(self._parent)


class _ChatCompletions:
    def __init__(self, parent: ThreadedAsyncOpenAIClient):
        self._parent = parent

    async def create(self, **kwargs) -> Any:
        return await self._parent._run_in_thread(lambda c: c.chat.completions.create(**kwargs))


class _Completions:
    def __init__(self, parent: ThreadedAsyncOpenAIClient):
        self._parent = parent

    async def create(self, **kwargs) -> Any:
        return await self._parent._run_in_thread(lambda c: c.completions.create(**kwargs))


class _Models:
    def __init__(self, parent: ThreadedAsyncOpenAIClient):
        self._parent = parent

    async def list(self) -> Any:
        return await self._parent._run_in_thread(lambda c: c.models.list())
