#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SpeechTimeoutUserTurnStopStrategy short-utterance short-circuit.

When STT emits a finalized=True transcript AND the accumulated text is
short (<= short_utterance_char_threshold), the strategy should fire
on_user_turn_stopped without waiting for the user_speech_timeout policy
floor.
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
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


# Use a deliberately LONG policy floor so we can tell short-circuit from full wait.
LONG_USER_SPEECH_TIMEOUT = 1.5
SHORT_CIRCUIT_BUDGET = 0.10  # what we believe a short-circuit completes in
STT_TIMEOUT = 0.0


class TestSpeechTimeoutShortUtteranceShortCircuit(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def _create_strategy(
        self,
        user_speech_timeout: float = LONG_USER_SPEECH_TIMEOUT,
        short_utterance_char_threshold: int = 20,
    ):
        strategy = SpeechTimeoutUserTurnStopStrategy(
            user_speech_timeout=user_speech_timeout,
            short_utterance_char_threshold=short_utterance_char_threshold,
        )
        await strategy.setup(self.task_manager)
        await strategy.process_frame(
            STTMetadataFrame(service_name="test", ttfs_p99_latency=STT_TIMEOUT)
        )
        return strategy

    async def test_short_utterance_finalized_short_circuits_under_policy_floor(self):
        """Short text + finalized=True should fire before the policy floor expires."""
        strategy = await self._create_strategy()

        fired_at: float | None = None
        start_time = time.monotonic()

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal fired_at
            fired_at = time.monotonic() - start_time

        # User speaks, then VAD says they stopped, then STT emits a short FINAL.
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await strategy.process_frame(
            TranscriptionFrame(
                text="ok", user_id="u", timestamp="", finalized=True
            )
        )

        # Give the event loop a beat to dispatch the handler.
        await asyncio.sleep(SHORT_CIRCUIT_BUDGET)

        self.assertIsNotNone(
            fired_at,
            f"on_user_turn_stopped never fired within {SHORT_CIRCUIT_BUDGET}s — "
            f"short-circuit did not happen",
        )
        # Must be well below the 1.5s policy floor.
        self.assertLess(
            fired_at,
            LONG_USER_SPEECH_TIMEOUT / 2,
            f"short-circuit took {fired_at:.3f}s; expected << "
            f"{LONG_USER_SPEECH_TIMEOUT}s policy floor",
        )

    async def test_long_utterance_finalized_does_NOT_short_circuit(self):
        """Long text + finalized=True should still wait for the policy floor."""
        strategy = await self._create_strategy()

        fired_at: float | None = None
        start_time = time.monotonic()

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal fired_at
            fired_at = time.monotonic() - start_time

        long_text = "long sentence with more than twenty characters here"
        assert len(long_text) > 20  # sanity check the fixture

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await strategy.process_frame(
            TranscriptionFrame(
                text=long_text, user_id="u", timestamp="", finalized=True
            )
        )

        # Wait a window that's WAY past short-circuit but well before the policy floor.
        await asyncio.sleep(SHORT_CIRCUIT_BUDGET * 3)

        self.assertIsNone(
            fired_at,
            f"on_user_turn_stopped fired at {fired_at}s — long utterance must "
            f"not short-circuit",
        )

        # Now wait for the policy floor to finish — it should fire then.
        await asyncio.sleep(LONG_USER_SPEECH_TIMEOUT + 0.1)
        self.assertIsNotNone(
            fired_at,
            "on_user_turn_stopped never fired even after the policy floor",
        )

    async def test_short_utterance_not_finalized_does_NOT_short_circuit(self):
        """Short text WITHOUT finalized=True should still wait for the policy floor."""
        strategy = await self._create_strategy()

        fired_at: float | None = None
        start_time = time.monotonic()

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal fired_at
            fired_at = time.monotonic() - start_time

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        # finalized defaults to False
        await strategy.process_frame(
            TranscriptionFrame(text="ok", user_id="u", timestamp="")
        )

        await asyncio.sleep(SHORT_CIRCUIT_BUDGET * 3)
        self.assertIsNone(
            fired_at,
            "non-finalized short transcript must not short-circuit",
        )

    async def test_threshold_zero_disables_short_circuit(self):
        """short_utterance_char_threshold=0 disables short-circuit entirely."""
        strategy = await self._create_strategy(short_utterance_char_threshold=0)

        fired_at: float | None = None
        start_time = time.monotonic()

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal fired_at
            fired_at = time.monotonic() - start_time

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await strategy.process_frame(
            TranscriptionFrame(
                text="ok", user_id="u", timestamp="", finalized=True
            )
        )

        await asyncio.sleep(SHORT_CIRCUIT_BUDGET * 3)
        self.assertIsNone(
            fired_at,
            "threshold=0 should disable short-circuit",
        )


if __name__ == "__main__":
    unittest.main()
