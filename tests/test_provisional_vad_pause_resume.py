#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Acceptance tests for ProvisionalVAD pause/resume (dograh 6937e01b → 63f0bc437).

Prod failure class this guards: a cough / noise / brief word during the bot's
greeting used to KILL the bot utterance (raw VAD barge-in cuts audio and
flushes buffers), leaving long silence. Contract:

1. Noise while the bot speaks: the VAD start only PAUSES output audio — no user
   turn, no interruption — and audio RESUMES after ``pause_secs`` with no
   transcript confirmation. The bot keeps talking.
2. Real speech while the bot speaks: a transcript inside the pause window
   resumes audio AND starts the user turn (the aggregator's InterruptionFrame
   is the cut).
3. When the bot is not speaking, VAD keeps the normal fast turn-start path.
"""

import asyncio
import unittest

from pipecat.frames.frames import (
    BotOutputAudioPauseFrame,
    BotOutputAudioResumeFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import ProvisionalVADUserTurnStartStrategy
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


class TestProvisionalVADPauseResumeAcceptance(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    def _make_strategy(self, pause_secs: float = 0.05):
        strategy = ProvisionalVADUserTurnStartStrategy(pause_secs=pause_secs)
        self.pushed_frames = []
        self.user_turns_started = 0

        @strategy.event_handler("on_push_frame")
        async def on_push_frame(strategy, frame, direction):
            self.pushed_frames.append(frame)

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            self.user_turns_started += 1

        return strategy

    def _frame_types(self):
        return [type(frame) for frame in self.pushed_frames]

    async def _wait_for_frame(self, frame_type, timeout: float = 1.0):
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if any(isinstance(frame, frame_type) for frame in self.pushed_frames):
                return True
            await asyncio.sleep(0.005)
        return False

    async def test_noise_during_bot_speech_pauses_then_resumes_without_cutting(self):
        """Noise (VAD, no transcript): pause ≤ pause_secs then RESUME, no turn."""
        strategy = self._make_strategy(pause_secs=0.05)
        await strategy.setup(self.task_manager)
        try:
            await strategy.process_frame(BotStartedSpeakingFrame())

            result = await strategy.process_frame(VADUserStartedSpeakingFrame())

            # No interruption, no user turn — just a pause.
            self.assertEqual(result, ProcessFrameResult.CONTINUE)
            self.assertEqual(self.user_turns_started, 0)
            self.assertEqual(self._frame_types(), [BotOutputAudioPauseFrame])

            # Window expires with no confirming transcript ⇒ audio resumes.
            self.assertTrue(await self._wait_for_frame(BotOutputAudioResumeFrame))
            self.assertEqual(
                self._frame_types(),
                [BotOutputAudioPauseFrame, BotOutputAudioResumeFrame],
            )
            # Still no cut: the bot keeps talking through the noise.
            self.assertEqual(self.user_turns_started, 0)
        finally:
            await strategy.cleanup()

    async def test_confirmed_speech_during_bot_speech_cuts(self):
        """A transcript inside the window: resume + user turn (the cut)."""
        strategy = self._make_strategy(pause_secs=1.0)
        await strategy.setup(self.task_manager)
        try:
            await strategy.process_frame(BotStartedSpeakingFrame())
            await strategy.process_frame(VADUserStartedSpeakingFrame())
            self.assertEqual(self._frame_types(), [BotOutputAudioPauseFrame])

            result = await strategy.process_frame(
                TranscriptionFrame(text="haan", user_id="cat", timestamp="")
            )

            # STOP ⇒ the controller starts the user turn ⇒ the aggregator
            # broadcasts the InterruptionFrame that cuts the bot audio.
            self.assertEqual(result, ProcessFrameResult.STOP)
            self.assertEqual(self.user_turns_started, 1)
            self.assertEqual(
                self._frame_types(),
                [BotOutputAudioPauseFrame, BotOutputAudioResumeFrame],
            )
        finally:
            await strategy.cleanup()

    async def test_interim_transcript_confirms_during_pause(self):
        """Interim transcripts (streaming STT) also confirm the turn."""
        strategy = self._make_strategy(pause_secs=1.0)
        await strategy.setup(self.task_manager)
        try:
            await strategy.process_frame(BotStartedSpeakingFrame())
            await strategy.process_frame(VADUserStartedSpeakingFrame())

            result = await strategy.process_frame(
                InterimTranscriptionFrame(text="haan", user_id="cat", timestamp="")
            )

            self.assertEqual(result, ProcessFrameResult.STOP)
            self.assertEqual(self.user_turns_started, 1)
        finally:
            await strategy.cleanup()

    async def test_vad_when_bot_not_speaking_starts_turn_immediately(self):
        """Fast path preserved: silence-side VAD starts the turn at once."""
        strategy = self._make_strategy(pause_secs=1.0)
        await strategy.setup(self.task_manager)
        try:
            result = await strategy.process_frame(VADUserStartedSpeakingFrame())

            self.assertEqual(result, ProcessFrameResult.STOP)
            self.assertEqual(self.user_turns_started, 1)
            # No pause frame: the bot was not speaking.
            self.assertNotIn(BotOutputAudioPauseFrame, self._frame_types())
        finally:
            await strategy.cleanup()

    async def test_late_transcript_after_resume_does_not_cut_until_bot_stops(self):
        """After the window expires, transcript starts are locked out until the
        bot stops speaking — a transcript cannot retro-cut the utterance."""
        strategy = self._make_strategy(pause_secs=0.02)
        await strategy.setup(self.task_manager)
        try:
            await strategy.process_frame(BotStartedSpeakingFrame())
            await strategy.process_frame(VADUserStartedSpeakingFrame())
            self.assertTrue(await self._wait_for_frame(BotOutputAudioResumeFrame))

            result = await strategy.process_frame(
                TranscriptionFrame(text="late", user_id="cat", timestamp="")
            )
            self.assertEqual(result, ProcessFrameResult.CONTINUE)
            self.assertEqual(self.user_turns_started, 0)

            # Once the bot stops, transcripts start turns again.
            await strategy.process_frame(BotStoppedSpeakingFrame())
            result = await strategy.process_frame(
                TranscriptionFrame(text="new", user_id="cat", timestamp="")
            )
            self.assertEqual(result, ProcessFrameResult.STOP)
            self.assertEqual(self.user_turns_started, 1)
        finally:
            await strategy.cleanup()
