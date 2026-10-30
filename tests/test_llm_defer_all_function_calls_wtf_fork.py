#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WTF fork: ``defer_function_calls_until_bot_stops`` restores the 1.1 tool-call timing.

Pipecat 1.12 waits for playback only for a response holding a single function
registered with ``is_node_transition=True``; every other call runs while the bot
is still speaking the same turn's text. WTF Voice ran on the Dograh 1.1 fork, which
deferred every call of a turn that produced speech until ``BotStoppedSpeakingFrame``,
so a "connecting you now" line finished before the transfer/end-call/workflow
transition happened. The flag (off by default, upstream behaviour unchanged) brings
that back:

- ordinary calls of a speaking turn run at BotStoppedSpeaking, interrupted or not
  (1.1 ran them at the interrupted stop too);
- they are dropped when a new completion starts before playback stops (1.1 reset
  its pending list on every new completion);
- calls of a turn without speech, or after playback already stopped, run at once;
- registered node transitions keep the upstream rules (dropped on interruption).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import BotStoppedSpeakingFrame, InterruptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.test_llm_transition_deferral import respond


@pytest_asyncio.fixture(params=["openai", "google"])
async def service(request):
    from unittest.mock import patch

    if request.param == "google":
        llm = GoogleLLMService(api_key="test-key")
    else:
        with patch.object(OpenAILLMService, "create_client"):
            llm = OpenAILLMService(api_key="test-key")
    await llm.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=TaskManager(),
            pipeline_worker=SimpleNamespace(app_resources=None, worker_runner=None),
        )
    )
    llm.push_frame = AsyncMock()
    llm.run_function_calls = AsyncMock()
    llm.register_function("end_call", AsyncMock(), is_node_transition=True)
    llm.register_function("transfer_call", AsyncMock())
    llm.register_function("save_booking", AsyncMock())
    llm.defer_function_calls_until_bot_stops = True
    yield llm
    await llm.cleanup()


def _ran(service) -> list[list[str]]:
    return [[c.function_name for c in call.args[0]] for call in service.run_function_calls.await_args_list]


@pytest.mark.asyncio
async def test_flag_is_off_by_default():
    from unittest.mock import patch

    with patch.object(OpenAILLMService, "create_client"):
        llm = OpenAILLMService(api_key="test-key")
    assert llm.defer_function_calls_until_bot_stops is False


@pytest.mark.asyncio
@pytest.mark.parametrize("names", [["transfer_call"], ["save_booking", "transfer_call"]])
async def test_ordinary_calls_of_a_speaking_turn_wait_for_playback(service, names):
    await respond(service, names)
    service.run_function_calls.assert_not_awaited()
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    assert _ran(service) == [names]
    # Released once only.
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    assert _ran(service) == [names]


@pytest.mark.asyncio
async def test_ordinary_calls_still_run_at_an_interrupted_stop(service):
    await respond(service, ["transfer_call"])
    await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
    stopped = BotStoppedSpeakingFrame()
    stopped.interrupted = True
    await service.process_frame(stopped, FrameDirection.UPSTREAM)
    assert _ran(service) == [["transfer_call"]]


@pytest.mark.asyncio
async def test_new_completion_before_playback_stops_drops_them(service):
    await respond(service, ["save_booking"])
    service.run_function_calls.assert_not_awaited()
    await respond(service, [], text="Sure, anything else?")
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    service.run_function_calls.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "...", "\n"])
async def test_calls_of_a_turn_without_speech_run_at_once(service, text):
    await respond(service, ["transfer_call"], text=text)
    assert _ran(service) == [["transfer_call"]]


@pytest.mark.asyncio
async def test_registered_transition_keeps_upstream_rules(service):
    # A lone transition waits for playback...
    await respond(service, ["end_call"])
    service.run_function_calls.assert_not_awaited()
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    assert _ran(service) == [["end_call"]]
    # ...and is dropped if the caller interrupts.
    service.run_function_calls.reset_mock()
    await respond(service, ["end_call"])
    await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    service.run_function_calls.assert_not_awaited()


@pytest.mark.asyncio
async def test_ordinary_and_transition_together(service):
    await respond(service, ["save_booking", "end_call"])
    service.run_function_calls.assert_not_awaited()
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    assert sorted(n for batch in _ran(service) for n in batch) == ["end_call", "save_booking"]
