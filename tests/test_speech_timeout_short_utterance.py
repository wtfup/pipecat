#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WTF fork: SpeechTimeoutUserTurnStopStrategy short-utterance short-circuit.

When STT emits a finalized=True transcript AND the accumulated text is short
(<= short_utterance_char_threshold), the strategy fires on_user_turn_stopped
without waiting out the user_speech_timeout policy floor. Ported from fork
commit 21a386a8f onto the Pipecat 1.12 strategy API.
"""

import asyncio
import time
import unittest

from pipecat.frames.frames import (
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup

# A deliberately LONG policy floor so a short-circuit is distinguishable from a full wait.
LONG_USER_SPEECH_TIMEOUT = 1.5
SHORT_CIRCUIT_BUDGET = 0.10
STT_TIMEOUT = 0.0


class TestSpeechTimeoutShortUtteranceShortCircuit(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()

    async def _create_strategy(self, short_utterance_char_threshold: int | None = None):
        kwargs = {"user_speech_timeout": LONG_USER_SPEECH_TIMEOUT}
        if short_utterance_char_threshold is not None:
            kwargs["short_utterance_char_threshold"] = short_utterance_char_threshold
        strategy = SpeechTimeoutUserTurnStopStrategy(**kwargs)
        await strategy.setup(frame_processor_setup(self.task_manager))
        await strategy.process_frame(
            STTMetadataFrame(service_name="test", ttfs_p99_latency=STT_TIMEOUT)
        )
        self.fired_at: float | None = None
        self.start_time = time.monotonic()

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            self.fired_at = time.monotonic() - self.start_time

        return strategy

    async def _speak(self, strategy, text: str, finalized: bool = True):
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await strategy.process_frame(
            TranscriptionFrame(text=text, user_id="u", timestamp="", finalized=finalized)
        )

    async def test_default_threshold_is_20(self):
        self.assertEqual(SpeechTimeoutUserTurnStopStrategy.SHORT_UTTERANCE_CHAR_THRESHOLD, 20)

    async def test_short_finalized_short_circuits_policy_floor(self):
        strategy = await self._create_strategy()
        await self._speak(strategy, "haan ji")
        await asyncio.sleep(SHORT_CIRCUIT_BUDGET)
        self.assertIsNotNone(self.fired_at, "short finalized reply did not short-circuit")
        self.assertLess(self.fired_at, LONG_USER_SPEECH_TIMEOUT / 2)
        await strategy.cleanup()

    async def test_long_finalized_waits_for_policy_floor(self):
        strategy = await self._create_strategy()
        await self._speak(strategy, "long sentence with more than twenty characters here")
        await asyncio.sleep(SHORT_CIRCUIT_BUDGET * 3)
        self.assertIsNone(self.fired_at, "a long utterance must not short-circuit")
        await asyncio.sleep(LONG_USER_SPEECH_TIMEOUT)
        self.assertIsNotNone(self.fired_at, "never fired even after the policy floor")
        await strategy.cleanup()

    async def test_short_not_finalized_waits(self):
        strategy = await self._create_strategy()
        await self._speak(strategy, "ok", finalized=False)
        await asyncio.sleep(SHORT_CIRCUIT_BUDGET * 3)
        self.assertIsNone(self.fired_at, "a non-finalized transcript must not short-circuit")
        await strategy.cleanup()

    async def test_threshold_zero_disables(self):
        strategy = await self._create_strategy(short_utterance_char_threshold=0)
        await self._speak(strategy, "ok")
        await asyncio.sleep(SHORT_CIRCUIT_BUDGET * 3)
        self.assertIsNone(self.fired_at, "threshold=0 must keep upstream behaviour")
        await strategy.cleanup()

    async def test_user_resumes_speaking_cancels_short_circuit(self):
        """A short final followed by more speech must not end the turn mid-sentence."""
        strategy = await self._create_strategy()
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        # Short final while VAD still reports speech: the turn must stay open.
        await strategy.process_frame(
            TranscriptionFrame(text="haan", user_id="u", timestamp="", finalized=True)
        )
        await asyncio.sleep(SHORT_CIRCUIT_BUDGET)
        self.assertIsNone(self.fired_at, "fired while VAD still reports the user speaking")
        await strategy.cleanup()


if __name__ == "__main__":
    unittest.main()
