import asyncio
import contextlib
import logging
from typing import cast

import aiohttp

from bioconda_utils.aiopipe import AsyncRequests as PipelineRequests
from bioconda_utils.conda.repodata import AsyncRequests as RepodataRequests
from bioconda_utils.support import http, logsetup


def test_make_session_uses_requested_user_agent():
    async def check():
        async with http.make_session(user_agent="custom-agent") as session:
            assert session.headers["User-Agent"] == "custom-agent"

    asyncio.run(check())


def test_pipeline_requests_preserves_user_agent_override():
    class CustomRequests(PipelineRequests):
        USER_AGENT = "custom-pipeline-agent"

    async def check():
        async with CustomRequests() as requests:
            assert requests.session is not None
            assert requests.session.headers["User-Agent"] == CustomRequests.USER_AGENT

    asyncio.run(check())


def test_repodata_requests_preserves_user_agent_override(monkeypatch):
    class CustomRequests(RepodataRequests):
        USER_AGENT = "custom-repodata-agent"

    original_make_session = http.make_session
    observed_user_agents = []

    def make_session(**kwargs):
        observed_user_agents.append(kwargs["user_agent"])
        return original_make_session(**kwargs)

    monkeypatch.setattr(http, "make_session", make_session)
    asyncio.run(CustomRequests.async_fetch([]))

    assert observed_user_agents == [CustomRequests.USER_AGENT]


def test_stream_download_yields_blocks_and_reports_progress(monkeypatch):
    class Content:
        def __init__(self):
            self.blocks = [b"first", b"second", b""]
            self.block_sizes = []

        async def read(self, block_size):
            self.block_sizes.append(block_size)
            return self.blocks.pop(0)

    class Response:
        def __init__(self):
            self.headers = {"Content-Length": "11"}
            self.content = Content()

    class Progress:
        def __init__(self):
            self.updates = []

        def update(self, size):
            self.updates.append(size)

    progress = Progress()
    progress_options = {}

    @contextlib.contextmanager
    def progress_factory(total=None, description=""):
        progress_options["total"] = total
        progress_options["description"] = description
        yield progress

    monkeypatch.setattr(http, "progress_bar", progress_factory)
    response = Response()

    async def download():
        return [
            block
            async for block in http.stream_download(
                cast(aiohttp.ClientResponse, response),
                "artifact",
                block_size=4,
            )
        ]

    assert asyncio.run(download()) == [b"first", b"second"]
    assert response.content.block_sizes == [4, 4, 4]
    assert progress.updates == [5, 6]
    assert progress_options["total"] == 11
    assert progress_options["description"] == "artifact"


def test_retry_policy_gives_up_only_on_permanent_response_errors():
    request_info = cast(aiohttp.RequestInfo, None)
    permanent = aiohttp.ClientResponseError(request_info, (), status=404)
    transient = aiohttp.ClientResponseError(request_info, (), status=503)

    assert http._give_up_on_http_error(permanent)
    assert not http._give_up_on_http_error(transient)
    assert not http._give_up_on_http_error(aiohttp.ClientPayloadError())


def test_progress_is_silent_when_redirected(monkeypatch):
    assert list(logsetup.track([1, 2], "x")) == [1, 2]

    with logsetup.progress_bar(total=2, description="x") as bar:
        bar.update(1)


def test_logger_treats_subprocess_output_as_literal_text():
    logger = logsetup.setup_logger("test-literal-logging", logging.INFO)

    # Rich markup would suppress the first value and raise MarkupError for the
    # second one. Logging arbitrary command output must never interpret either.
    logger.info("[not-a-style]")
    logger.info("unmatched closing tag [/bold]")
