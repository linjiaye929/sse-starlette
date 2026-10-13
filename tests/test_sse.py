import asyncio
import logging
import math
from functools import partial

import anyio
import anyio.lowlevel
import pytest
from starlette.background import BackgroundTask
from starlette.testclient import TestClient

from sse_starlette.event import ServerSentEvent
from sse_starlette.sse import EventSourceResponse
from sse_starlette.sse import SendTimeoutError
from tests.anyio_compat import collapse_excgroups

_log = logging.getLogger(__name__)


# Test fixtures and helpers
@pytest.fixture
def mock_generator():
    async def numbers(minimum, maximum):
        for i in range(minimum, maximum + 1):
            await asyncio.sleep(0.1)
            yield i

    return numbers


@pytest.fixture
def mock_memory_channels():
    async def setup():
        send_chan, recv_chan = anyio.create_memory_object_stream(math.inf)
        return send_chan, recv_chan

    return setup


class TestEventSourceResponse:
    @pytest.mark.parametrize(
        "input_type,separator,expected_output",
        [
            ("integer", "\r\n", b"data: 1\r\n\r\n"),
            ("dict_simple", "\r\n", b"data: 1\r\n\r\n"),
            ("dict_with_event", "\r\n", b"event: message\r\ndata: 1\r\n\r\n"),
            ("dict_with_event", "\r", b"event: message\rdata: 1\r\r"),
        ],
    )
    async def test_response_send_whenValidInput_thenGeneratesExpectedOutput(
        self,
        mock_generator,
        input_type,
        separator,
        expected_output,
    ):
        # Arrange
        async def app(scope, receive, send):
            async def format_output(value):
                if input_type == "integer":
                    return value
                elif input_type == "dict_simple":
                    return dict(data=value)
                else:
                    return dict(data=value, event="message")

            async def generate():
                generator = mock_generator(1, 5)
                async for value in generator:
                    yield await format_output(value)

            # ping=0: keep serialization assertions independent of heartbeat timing
            response = EventSourceResponse(generate(), sep=separator, ping=0)
            await response(scope, receive, send)

        # Act
        client = TestClient(app)
        response = client.get("/")

        # Assert
        assert expected_output in response.content

    @pytest.mark.parametrize(
        "producer_output,expected_sse_response",
        [
            # Test raw integers being converted to SSE format
            ("raw_integer", b"data: 1\r\n\r\n"),
            # Test dict with just data field
            ("simple_dict", b"data: 1\r\n\r\n"),
            # Test dict with both event and data fields
            ("event_dict", b"event: message\r\ndata: 1\r\n\r\n"),
        ],
    )
    def test_eventSourceResponse_whenUsingMemoryChannel_thenHandlesAsyncQueueCorrectly(
        self, producer_output, expected_sse_response
    ):
        """Tests that EventSourceResponse can properly consume data from an async memory channel.

        This test verifies the producer-consumer pattern where:
        1. Producer (stream_numbers) puts data into a memory channel
        2. Consumer (EventSourceResponse) reads from that channel and formats as SSE

        This differs from direct generator tests by:
        - Using separate producer/consumer components
        - Testing async queue-based communication
        - Verifying SSE works with buffered async data sources
        """

        # Arrange
        async def app(scope, receive, send):
            # Create bounded memory channel for producer-consumer communication
            send_chan, recv_chan = anyio.create_memory_object_stream(
                max_buffer_size=math.inf
            )

            # Producer function that writes to the channel
            async def stream_numbers(producer_channel, start, end):
                async with producer_channel:
                    for i in range(start, end + 1):
                        await anyio.sleep(0.1)  # Simulate async data production

                        # Format data based on test case
                        if producer_output == "raw_integer":
                            data = i
                        elif producer_output == "simple_dict":
                            data = dict(data=i)
                        else:  # event_dict
                            data = dict(data=i, event="message")

                        # Send to channel for consumption
                        await producer_channel.send(data)

            # Create SSE response that consumes from channel
            response = EventSourceResponse(
                recv_chan,  # Consumer reads from receive channel
                data_sender_callable=partial(
                    stream_numbers, send_chan, 1, 5
                ),  # Producer writes to send channel
                ping=0,  # keep assertions independent of heartbeat timing
            )
            await response(scope, receive, send)

        # Act
        client = TestClient(app)
        response = client.get("/")

        # Assert
        assert expected_sse_response in response.content

    @pytest.mark.anyio
    async def test_disconnect_whenClientDisconnects_thenHandlesGracefully(
        self, httpx_client, caplog
    ):
        # Arrange
        caplog.set_level(logging.DEBUG)

        # Act & Assert
        with pytest.raises(TimeoutError):
            with anyio.fail_after(1) as scope:
                try:
                    async with anyio.create_task_group() as tg:
                        # https://www.python-httpx.org/async/#streaming-responses
                        tg.start_soon(httpx_client.get, "/endless")
                finally:
                    assert scope.cancel_called is True
                    assert "chunk: b'data: 4\\r\\n\\r\\n'" in caplog.text
                    assert "Disconnected from client" in caplog.text

    @pytest.mark.anyio
    @pytest.mark.parametrize("async_cleanup", [False, True])
    async def test_disconnect_whenDataSendBlocked_thenClosesStartedGenerator(
        self, async_cleanup
    ):
        send_started = anyio.Event()
        cleanup_count = 0

        async def event_publisher():
            nonlocal cleanup_count
            try:
                yield {"data": "first event"}
                pytest.fail("Disconnected streams must not resume iteration")
            finally:
                if async_cleanup:
                    await anyio.lowlevel.checkpoint()
                cleanup_count += 1

        async def send(message):
            if message["type"] == "http.response.body":
                send_started.set()
                await anyio.sleep_forever()

        async def receive():
            await send_started.wait()
            return {"type": "http.disconnect"}

        # Keep both references alive: cleanup must not depend on garbage collection.
        generator = event_publisher()
        response = EventSourceResponse(generator, ping=0)
        try:
            with anyio.fail_after(5):
                await response(
                    {"type": "http", "method": "GET", "headers": []}, receive, send
                )
            assert cleanup_count == 1
        finally:
            await generator.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "termination", ["disconnect", "timeout", "send_error", "exhausted", "next"]
    )
    async def test_streamCleanup_whenTerminated_thenPreservesIteratorOwnership(
        self, termination
    ):
        send_started = anyio.Event()
        next_started = anyio.Event()
        send_cancellation = None
        stream_error = None
        send_error = OSError("send failed")

        class Iterator:
            next_calls = 0
            close_calls = 0
            cancelled_next = 0
            running = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                self.running = True
                self.next_calls += 1
                try:
                    if self.next_calls == 1:
                        return {"data": "first event"}
                    if termination == "next":
                        next_started.set()
                        try:
                            await anyio.sleep_forever()
                        except anyio.get_cancelled_exc_class():
                            self.cancelled_next += 1
                            raise
                    raise StopAsyncIteration
                finally:
                    self.running = False

            async def aclose(self):
                assert not self.running, "Never close a running iterator"
                self.close_calls += 1
                assert self.close_calls == 1, "Never close the iterator twice"
                await anyio.lowlevel.checkpoint()

        class RecordingResponse(EventSourceResponse):
            async def _stream_response(self, send):
                nonlocal stream_error
                try:
                    await super()._stream_response(send)
                except BaseException as exc:
                    stream_error = exc
                    raise

        async def send(message):
            nonlocal send_cancellation
            if message["type"] != "http.response.body" or not message["more_body"]:
                return
            send_started.set()
            if termination == "send_error":
                raise send_error
            if termination in ("disconnect", "timeout"):
                try:
                    await anyio.sleep_forever()
                except anyio.get_cancelled_exc_class() as exc:
                    send_cancellation = exc
                    raise

        async def receive():
            if termination == "disconnect":
                await send_started.wait()
            elif termination == "next":
                await next_started.wait()
            else:
                await anyio.sleep_forever()
            return {"type": "http.disconnect"}

        iterator = Iterator()
        response = RecordingResponse(
            iterator, ping=0, send_timeout=0.01 if termination == "timeout" else None
        )
        scope = {"type": "http", "method": "GET", "headers": []}
        with anyio.fail_after(5):
            if termination in ("timeout", "send_error"):
                error_type = SendTimeoutError if termination == "timeout" else OSError
                with pytest.raises(error_type) as exc_info:
                    await response(scope, receive, send)
                assert exc_info.value is stream_error
                if termination == "send_error":
                    assert exc_info.value is send_error
            else:
                await response(scope, receive, send)

        assert iterator.close_calls == int(termination in ("disconnect", "timeout"))
        assert iterator.cancelled_next == int(termination == "next")
        if termination == "disconnect":
            assert stream_error is send_cancellation

    @pytest.mark.anyio
    async def test_sendTimeout_whenCleanupCancelled_thenHonorsOuterCancellation(self):
        close_started = anyio.Event()
        release_close = anyio.Event()
        response_finished = anyio.Event()
        cleanup_cancelled = False
        cleanup_finished = False
        response_errors = []

        async def event_publisher():
            nonlocal cleanup_cancelled, cleanup_finished
            try:
                yield {"data": "first event"}
            finally:
                close_started.set()
                try:
                    await release_close.wait()
                except anyio.get_cancelled_exc_class():
                    cleanup_cancelled = True
                    raise
                cleanup_finished = True

        async def send(message):
            if message["type"] == "http.response.body":
                await anyio.sleep_forever()

        async def receive():
            await anyio.sleep_forever()

        generator = event_publisher()
        response = EventSourceResponse(generator, ping=0, send_timeout=0.01)

        async def run_response(task_status=anyio.TASK_STATUS_IGNORED):
            with anyio.CancelScope() as cancel_scope:
                task_status.started(cancel_scope)
                try:
                    await response(
                        {"type": "http", "method": "GET", "headers": []},
                        receive,
                        send,
                    )
                except BaseException as exc:
                    response_errors.append(exc)
            response_finished.set()

        async with anyio.create_task_group() as task_group:
            try:
                cancel_scope = await task_group.start(run_response)
                # The watchdog bounds a broken implementation, not an elapsed-time
                # assertion: cancellation must finish before cleanup is released.
                with anyio.fail_after(5):
                    await close_started.wait()
                    cancel_scope.cancel()
                    await response_finished.wait()
                assert cleanup_cancelled
                assert not cleanup_finished
                assert not release_close.is_set()
                assert len(response_errors) == 1
                assert isinstance(response_errors[0], anyio.get_cancelled_exc_class())
            finally:
                release_close.set()
        await generator.aclose()

    @pytest.mark.anyio
    async def test_send_whenTimeoutOccurs_thenRaisesSendTimeoutError(self):
        # Arrange
        # Send timeout is set to 0.5s, but `send` will take 1s. Expect SendTimeoutError.
        cleanup_executed = False

        async def event_publisher():
            try:
                yield {"event": "test", "data": "data"}
                pytest.fail("Should not reach this point")
            finally:
                nonlocal cleanup_executed
                cleanup_executed = True

        async def mock_send(*args, **kwargs):
            await anyio.sleep(1.0)

        async def mock_receive():
            await anyio.lowlevel.checkpoint()
            return {"type": "message"}

        response = EventSourceResponse(event_publisher(), send_timeout=0.5)

        # Act & Assert
        with pytest.raises(SendTimeoutError):
            with collapse_excgroups():
                await response(
                    {"type": "http", "method": "GET", "headers": []},
                    mock_receive,
                    mock_send,
                )

        assert cleanup_executed, "Cleanup should be executed on timeout"

    def test_headers_whenCustomHeadersProvided_thenMergesCorrectly(self):
        # Arrange
        custom_headers = {
            "cache-control": "no-cache",
            "x-accel-buffering": "yes",  # Should not override
            "connection": "close",  # Should not override
            "x-custom-header": "custom-value",
        }

        # Act
        response = EventSourceResponse(range(1, 5), headers=custom_headers, ping=0.2)
        headers = dict((h.decode(), v.decode()) for h, v in response.raw_headers)

        # Assert
        assert headers["cache-control"] == "no-cache"
        assert headers["x-accel-buffering"] == "no"  # Should keep default
        assert headers["connection"] == "keep-alive"  # Should keep default
        assert headers["x-custom-header"] == "custom-value"
        assert headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_headers_whenCreated_thenHasCorrectCharset(self, mock_generator):
        # Arrange
        generator = mock_generator(1, 5)

        # Act
        response = EventSourceResponse(generator, ping=0.2)
        content_type_headers = [
            (h.decode(), v.decode())
            for h, v in response.raw_headers
            if h.decode() == "content-type"
        ]

        # Assert
        assert len(content_type_headers) == 1
        header_name, header_value = content_type_headers[0]
        assert header_value == "text/event-stream; charset=utf-8"

    @pytest.mark.anyio
    async def test_ping_whenTimerTicks_thenSendsOneMessagePerTick(self, monkeypatch):
        """Test ping cadence with logical ticks instead of elapsed wall time.

        The fake sleep records every requested interval and yields at a
        cooperative checkpoint so cancellation and task scheduling semantics are
        preserved without making the assertion depend on machine load. Two ticks
        verify that ping delivery is periodic rather than a one-shot operation.
        """

        # Arrange
        ping_interval = 0.2
        expected_ping_count = 2
        sleep_intervals = []
        sent_messages = []

        async def counted_sleep(interval):
            sleep_intervals.append(interval)
            assert len(sleep_intervals) <= expected_ping_count
            await anyio.lowlevel.checkpoint()

        response = EventSourceResponse(
            [],
            ping=ping_interval,
            ping_message_factory=lambda: ServerSentEvent(comment="ping"),
        )

        async def send(message):
            sent_messages.append(message)
            if len(sent_messages) == expected_ping_count:
                response.active = False

        monkeypatch.setattr("sse_starlette.sse.anyio.sleep", counted_sleep)

        # Act
        await response._ping(send)

        # Assert
        assert sleep_intervals == [ping_interval] * expected_ping_count
        expected_ping_message = {
            "type": "http.response.body",
            "body": b": ping\r\n\r\n",
            "more_body": True,
        }
        assert sent_messages == [expected_ping_message] * expected_ping_count

    @pytest.mark.anyio
    async def test_ping_whenSendTimesOut_thenRaisesSendTimeoutError(self, monkeypatch):
        """Heartbeats must honor the response's per-send timeout.

        The ping interval is advanced by a logical tick, following
        test_ping_whenTimerTicks_thenSendsOneMessagePerTick, so send_timeout is
        the only real timer in play and the result does not depend on machine
        load. fail_after is a hang watchdog, not an assertion, so it is set
        generously.
        """

        # Arrange
        send_timeout = 0.05

        async def instant_sleep(_interval):
            await anyio.lowlevel.checkpoint()

        monkeypatch.setattr("sse_starlette.sse.anyio.sleep", instant_sleep)

        response = EventSourceResponse([], ping=10, send_timeout=send_timeout)

        async def blocked_send(_message):
            await anyio.sleep_forever()

        # Act & Assert
        with anyio.fail_after(5):
            with pytest.raises(SendTimeoutError):
                await response._ping(blocked_send)

    @pytest.mark.anyio
    async def test_ping_whenSendTimesOut_thenTearsDownResponseAndRunsCleanup(self):
        """A heartbeat timeout must tear down the whole response, not just _ping.

        This is the frozen-client case the send timeout exists for: the peer keeps
        the socket open but stops reading, so no http.disconnect ever arrives and
        the heartbeat is the only traffic. Unlike the unit test above, this drives
        the full ASGI entrypoint and asserts the generator's cleanup still runs.
        """

        # Arrange
        cleanup_executed = False

        async def event_publisher():
            try:
                await anyio.sleep_forever()
                yield {"event": "test", "data": "data"}
            finally:
                nonlocal cleanup_executed
                cleanup_executed = True

        async def mock_send(message):
            if message["type"] == "http.response.start":
                return
            await anyio.sleep_forever()

        async def mock_receive():
            await anyio.sleep_forever()

        response = EventSourceResponse(event_publisher(), ping=0.05, send_timeout=0.05)

        # Act & Assert
        with anyio.fail_after(5):
            with pytest.raises(SendTimeoutError):
                with collapse_excgroups():
                    await response(
                        {"type": "http", "method": "GET", "headers": []},
                        mock_receive,
                        mock_send,
                    )

        assert cleanup_executed, "Generator cleanup must run when a heartbeat times out"

    @pytest.mark.anyio
    async def test_ping_whenConcurrentWithEvents_thenRespectsLocking(self):
        # Sequencing here is as follows to reproduce race condition:
        # t=0.5s - event_publisher sends the first response item,
        #          claiming the lock and going to sleep for 1 second so until t=1.5s.
        # t=1.0s - ping task wakes up and tries to call send while we know
        #          that event_publisher is still blocked inside it and holding the lock
        # Arrange
        lock = anyio.Lock()

        async def event_publisher():
            for i in range(2):
                await anyio.sleep(0.5)
                yield i

        async def send(*args, **kwargs):
            # Raises WouldBlock if called while someone else already holds the lock
            lock.acquire_nowait()
            await anyio.sleep(1.0)
            lock.release()

        async def receive():
            await anyio.lowlevel.checkpoint()
            return {"type": "message"}

        response = EventSourceResponse(event_publisher(), ping=1)

        # Act & Assert
        with pytest.raises(anyio.WouldBlock):
            with collapse_excgroups():
                await response(
                    {"type": "http", "method": "GET", "headers": []}, receive, send
                )

    def test_pingInterval_whenCreated_thenUsesDefaultValue(self):
        # Arrange & Act
        response = EventSourceResponse(0)

        # Assert
        assert response.ping_interval == response.DEFAULT_PING_INTERVAL

    def test_pingInterval_whenValidValueSet_thenUpdatesInterval(self):
        # Arrange
        response = EventSourceResponse(0)
        new_interval = 25

        # Act
        response.ping_interval = new_interval

        # Assert
        assert response.ping_interval == new_interval

    def test_pingInterval_whenStringProvided_thenRaisesTypeError(self):
        # Arrange
        response = EventSourceResponse(0)
        invalid_interval = "ten"

        # Act & Assert
        with pytest.raises(TypeError, match="ping interval must be int"):
            response.ping_interval = invalid_interval

    def test_pingInterval_whenNegativeValue_thenRaisesValueError(self):
        # Arrange
        response = EventSourceResponse(0)
        negative_interval = -42

        # Act & Assert
        with pytest.raises(ValueError, match=r"ping interval must be >= 0"):
            response.ping_interval = negative_interval

    def test_pingInterval_whenZero_thenAcceptedAsDisabled(self):
        # Arrange
        response = EventSourceResponse(0)

        # Act
        response.ping_interval = 0

        # Assert
        assert response.ping_interval == 0

    @pytest.mark.anyio
    async def test_ping_whenIntervalIsZero_thenNoPingTaskIsStarted(self):
        """Issue #206: ping=0 must disable pings, not busy-spin on sleep(0).

        Before the fix, ``_ping`` looped on ``anyio.sleep(0)`` — a bare
        checkpoint — emitting tens of thousands of pings per second and
        contending for ``_send_lock`` with the data stream.
        """

        # Arrange
        async def gen():
            for i in range(3):
                await anyio.sleep(0.01)
                yield {"data": i}

        response = EventSourceResponse(gen(), ping=0)
        bodies = []

        async def send(message):
            if message["type"] == "http.response.body":
                bodies.append(message["body"])

        async def receive():
            await anyio.sleep(math.inf)
            return {"type": "http.disconnect"}  # pragma: no cover

        # Act
        await response({"type": "http", "method": "GET", "headers": []}, receive, send)

        # Assert
        assert not any(b": ping" in body for body in bodies)
        assert [b for b in bodies if b] == [
            b"data: 0\r\n\r\n",
            b"data: 1\r\n\r\n",
            b"data: 2\r\n\r\n",
        ]

    def test_compression_whenEnabled_thenRaisesNotImplemented(self):
        # Arrange
        response = EventSourceResponse(range(1, 5))

        # Act & Assert
        with pytest.raises(NotImplementedError):
            response.enable_compression()

    @pytest.mark.anyio
    async def test_backgroundTask_whenProvided_thenExecutesAfterResponse(self):
        # Arrange
        task_executed = False

        async def background_task():
            nonlocal task_executed
            task_executed = True

        async def mock_send(*args, **kwargs):
            pass

        async def mock_receive():
            await anyio.lowlevel.checkpoint()
            return {"type": "http.disconnect"}

        response = EventSourceResponse([], background=BackgroundTask(background_task))

        # Act
        await response(
            {"type": "http", "method": "GET", "headers": []}, mock_receive, mock_send
        )

        # Assert
        assert task_executed, "Background task should be executed"
