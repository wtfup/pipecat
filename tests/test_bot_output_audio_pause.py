#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pause/resume output-audio transport tests (dograh 63f0bc437 port).

Adapted for our pin from the upstream ``tests/test_base_output_transport.py``
that shipped with 63f0bc437:

- ``pipeline_worker=`` → ``pipeline_task=`` (our pin's ``FrameProcessorSetup``
  field name).
- Only the pause/resume-relevant tests are carried. The two upstream-era tests
  that assert post-pin transport behaviour we don't have
  (``test_non_audio_frame_does_not_reset_consecutive_audio_write_failures`` and
  ``test_interruption_with_mixer_keeps_audio_task_and_mixer_output`` — upstream
  changes from June 2026; our pin's ``handle_interruptions`` still recreates the
  audio task and the failure counter resets on any frame) are intentionally NOT
  ported. Verified at 63f0bc437 itself: all 7 upstream tests pass there, so the
  gap is pin drift, not port damage.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    BotOutputAudioPauseFrame,
    BotOutputAudioResumeFrame,
    CancelFrame,
    EndFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


class TestBotOutputAudioPause(unittest.IsolatedAsyncioTestCase):
    async def _make_transport(self, **param_kwargs) -> BaseOutputTransport:
        params = TransportParams(audio_out_enabled=True, **param_kwargs)
        transport = BaseOutputTransport(params)
        transport.push_frame = AsyncMock()
        transport.write_audio_frame = AsyncMock(return_value=True)

        task_manager = TaskManager()
        task_manager.setup(TaskManagerParams(loop=asyncio.get_event_loop()))
        await transport.setup(
            FrameProcessorSetup(
                clock=SystemClock(),
                task_manager=task_manager,
                pipeline_task=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
            )
        )
        start_frame = StartFrame(audio_out_sample_rate=16000)
        await transport.process_frame(start_frame, FrameDirection.DOWNSTREAM)
        await transport.set_transport_ready(start_frame)
        return transport

    async def test_audio_pause_preserves_queued_bot_audio_until_resume(self):
        transport = await self._make_transport()
        try:
            sender = transport._media_senders[None]
            write_started = asyncio.Event()
            release_write = asyncio.Event()
            written_audio: list[bytes] = []

            async def slow_first_write(frame):
                written_audio.append(frame.audio)
                if len(written_audio) == 1:
                    write_started.set()
                    await release_write.wait()
                return True

            transport.write_audio_frame = AsyncMock(side_effect=slow_first_write)

            audio_one = OutputAudioRawFrame(
                audio=b"\x01\x02" * (sender.audio_chunk_size // 2),
                sample_rate=sender.sample_rate,
                num_channels=1,
            )
            await transport.process_frame(audio_one, FrameDirection.DOWNSTREAM)
            await write_started.wait()

            await transport.process_frame(BotOutputAudioPauseFrame(), FrameDirection.DOWNSTREAM)

            audio_two = OutputAudioRawFrame(
                audio=b"\x03\x04" * (sender.audio_chunk_size // 2),
                sample_rate=sender.sample_rate,
                num_channels=1,
            )
            await transport.process_frame(audio_two, FrameDirection.DOWNSTREAM)

            release_write.set()
            await asyncio.sleep(0.05)

            self.assertEqual(len(written_audio), 1)
            self.assertFalse(sender._audio_queue.empty())

            await transport.process_frame(BotOutputAudioResumeFrame(), FrameDirection.DOWNSTREAM)

            for _ in range(20):
                if len(written_audio) >= 2:
                    break
                await asyncio.sleep(0.01)

            self.assertEqual(len(written_audio), 2)
            self.assertEqual(written_audio[1], audio_two.audio)
        finally:
            await transport.cancel(CancelFrame())

    async def test_audio_pause_holds_audio_queued_after_pause(self):
        transport = await self._make_transport()
        try:
            sender = transport._media_senders[None]
            written_audio: list[bytes] = []

            async def capture_write(frame):
                written_audio.append(frame.audio)
                return True

            transport.write_audio_frame = AsyncMock(side_effect=capture_write)

            await transport.process_frame(BotOutputAudioPauseFrame(), FrameDirection.DOWNSTREAM)

            audio = OutputAudioRawFrame(
                audio=b"\x05\x06" * (sender.audio_chunk_size // 2),
                sample_rate=sender.sample_rate,
                num_channels=1,
            )
            await transport.process_frame(audio, FrameDirection.DOWNSTREAM)
            await asyncio.sleep(0.05)

            self.assertEqual(written_audio, [])

            await transport.process_frame(BotOutputAudioResumeFrame(), FrameDirection.DOWNSTREAM)

            for _ in range(20):
                if written_audio:
                    break
                await asyncio.sleep(0.01)

            self.assertEqual(written_audio, [audio.audio])
        finally:
            await transport.cancel(CancelFrame())

    async def test_stop_while_paused_drains_queue_without_deadlock(self):
        # A graceful EndFrame shutdown while output audio is paused must not
        # hang: stop() resumes the audio task so it can drain to the EndFrame.
        transport = await self._make_transport(audio_out_end_silence_secs=0)
        stopped = False
        try:
            sender = transport._media_senders[None]
            written_audio: list[bytes] = []

            async def capture_write(frame):
                written_audio.append(frame.audio)
                return True

            transport.write_audio_frame = AsyncMock(side_effect=capture_write)

            await transport.process_frame(BotOutputAudioPauseFrame(), FrameDirection.DOWNSTREAM)

            audio = OutputAudioRawFrame(
                audio=b"\x07\x08" * (sender.audio_chunk_size // 2),
                sample_rate=sender.sample_rate,
                num_channels=1,
            )
            await transport.process_frame(audio, FrameDirection.DOWNSTREAM)

            # Without the resume in stop(), the audio task stays blocked on the
            # pause event, never reads the EndFrame, and this awaits forever.
            await asyncio.wait_for(sender.stop(EndFrame()), timeout=5.0)
            stopped = True

            # The frame queued before shutdown is still delivered.
            self.assertEqual(written_audio, [audio.audio])
        finally:
            if not stopped:
                await transport.cancel(CancelFrame())

    async def test_interruption_while_paused_recovers_and_does_not_deadlock(self):
        # A barge-in while output audio is paused must unstick the transport:
        # handle_interruptions resumes audio before the audio task is replaced,
        # so post-cut audio flows again (without the resume, the recreated task
        # would wake straight into the cleared resume event and never write —
        # i.e. permanent bot silence). Note: a chunk the suspended reader had
        # already reclaimed may still be written once (upstream behaviour);
        # everything still queue-resident is discarded with the queue swap.
        transport = await self._make_transport()
        try:
            sender = transport._media_senders[None]
            written_audio: list[bytes] = []

            async def capture_write(frame):
                written_audio.append(frame.audio)
                return True

            transport.write_audio_frame = AsyncMock(side_effect=capture_write)

            await transport.process_frame(BotOutputAudioPauseFrame(), FrameDirection.DOWNSTREAM)

            stale_audio = OutputAudioRawFrame(
                audio=b"\x0a\x0b" * (sender.audio_chunk_size // 2),
                sample_rate=sender.sample_rate,
                num_channels=1,
            )
            await transport.process_frame(stale_audio, FrameDirection.DOWNSTREAM)
            await asyncio.sleep(0.05)
            self.assertEqual(written_audio, [])

            await transport.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

            # The pause must be cleared by the interruption path.
            self.assertFalse(sender._audio_paused)

            new_audio = OutputAudioRawFrame(
                audio=b"\x0c\x0d" * (sender.audio_chunk_size // 2),
                sample_rate=sender.sample_rate,
                num_channels=1,
            )
            await transport.process_frame(new_audio, FrameDirection.DOWNSTREAM)

            for _ in range(50):
                if new_audio.audio in written_audio:
                    break
                await asyncio.sleep(0.01)

            self.assertIn(new_audio.audio, written_audio)
        finally:
            await transport.cancel(CancelFrame())
