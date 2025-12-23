import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

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
        max_workers: int,
        timeout: int = 1200,
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
        self._max_retries = max_retries
        self._headers = headers or {}
        self._tls_key = f"oai_client_{id(self)}"

    @property
    def base_url(self) -> str:
        return self._base_url

    def _create_client(self) -> AsyncOpenAI:
        timeout = httpx.Timeout(self._timeout)
        http_client = httpx.AsyncClient(timeout=timeout, headers=self._headers)
        return AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            max_retries=self._max_retries,
            http_client=http_client,
        )

    def _get_thread_client(self) -> AsyncOpenAI:
        return get_or_create_thread_attr(self._tls_key, self._create_client)

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """Dynamically proxy attribute access to dispatch method calls to the thread pool."""
        outer = self

        class _ChainedMethodPatch:
            """Walk the path to the method and call it in the thread pool."""

            def __init__(self, path: tuple[str, ...]):
                self.path = path

            def __getattr__(self, attr_name: str) -> "_ChainedMethodPatch":
                return _ChainedMethodPatch(self.path + (attr_name,))

            async def __call__(self, *args, **kwargs):
                def run_in_thread():
                    loop = get_or_create_thread_loop()
                    client = outer._get_thread_client()
                    method = client
                    for attr in self.path:
                        method = getattr(method, attr)
                    return loop.run_until_complete(method(*args, **kwargs))

                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(outer.executor, run_in_thread)

        return _ChainedMethodPatch((name,))

    def teardown(self, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait)
