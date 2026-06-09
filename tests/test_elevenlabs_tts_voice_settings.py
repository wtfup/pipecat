#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for ElevenLabs TTS WebSocket ``voice_settings`` consistency.

ElevenLabs' ``multi-stream-input`` endpoint requires ``voice_settings`` to be
provided in the *first message of the socket session* and then either omitted
or unchanged. Re-sending ``voice_settings`` on a later context (a new turn)
closes the socket with WS 1008. Because dograh runs the service with
``reconnect_on_error=False``, that kills TTS for the rest of the call — the
"works for the greeting, dies ~40s in" signature observed on live runs 27/28.

These tests pin the contract: ``voice_settings`` (and
``pronunciation_dictionary_locators``) ride the FIRST context-init message of a
socket and are never re-sent on the same socket. A voice-settings change forces
a full reconnect (so the new value legally rides the next socket's first
message) rather than an in-socket resend.
"""

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from websockets.protocol import State

from pipecat.services.elevenlabs.tts import (
    ElevenLabsTTSService,
    ElevenLabsTTSSettings,
)


class _FakeWebsocket:
    """Records every payload sent over the socket as a parsed dict."""

    def __init__(self):
        self.sent: list[dict[str, Any]] = []
        self.state = State.OPEN

    async def send(self, payload: str):
        self.sent.append(json.loads(payload))

    async def close(self):
        pass


def _make_service(**settings_kwargs) -> ElevenLabsTTSService:
    settings = ElevenLabsTTSSettings(
        voice="test-voice",
        model="eleven_turbo_v2_5",
        **settings_kwargs,
    )
    service = ElevenLabsTTSService(
        api_key="test-key",
        settings=settings,
    )
    return service


async def _drive_run_tts(service: ElevenLabsTTSService, text: str, context_id: str):
    """Drive run_tts for a single (text, context_id), draining the generator.

    The audio-context bookkeeping and metrics are stubbed so the test can run
    without a live pipeline / task manager.
    """
    service.create_audio_context = AsyncMock()
    service.start_ttfb_metrics = AsyncMock()
    service.start_tts_usage_metrics = AsyncMock()
    # Force the "new context" branch every time.
    service.audio_context_available = lambda cid: False

    async for _ in service.run_tts(text, context_id):
        pass


def _voice_settings_messages(ws: _FakeWebsocket) -> list[dict[str, Any]]:
    return [m for m in ws.sent if "voice_settings" in m]


@pytest.mark.asyncio
async def test_voice_settings_only_on_first_context():
    service = _make_service(stability=0.8, similarity_boost=0.75)
    ws = _FakeWebsocket()
    service._websocket = ws
    # Mark socket as freshly connected (no settings sent yet).
    service._voice_settings_sent = False

    await _drive_run_tts(service, "hello", "ctx-1")
    await _drive_run_tts(service, "world", "ctx-2")

    init_msgs = [m for m in ws.sent if m.get("text") == " "]
    assert len(init_msgs) == 2
    assert "voice_settings" in init_msgs[0]
    assert "voice_settings" not in init_msgs[1]


@pytest.mark.asyncio
async def test_voice_settings_sent_exactly_once_across_n_contexts():
    service = _make_service(stability=0.8, similarity_boost=0.75)
    ws = _FakeWebsocket()
    service._websocket = ws
    service._voice_settings_sent = False

    for i in range(5):
        await _drive_run_tts(service, f"chunk {i}", f"ctx-{i}")

    assert len(_voice_settings_messages(ws)) == 1


@pytest.mark.asyncio
async def test_pronunciation_locators_only_on_first_context():
    from pipecat.services.elevenlabs.tts import PronunciationDictionaryLocator

    service = _make_service(stability=0.8)
    service._pronunciation_dictionary_locators = [
        PronunciationDictionaryLocator(
            pronunciation_dictionary_id="dict-1", version_id="v1"
        )
    ]
    ws = _FakeWebsocket()
    service._websocket = ws
    service._voice_settings_sent = False

    await _drive_run_tts(service, "hello", "ctx-1")
    await _drive_run_tts(service, "world", "ctx-2")

    locator_msgs = [m for m in ws.sent if "pronunciation_dictionary_locators" in m]
    assert len(locator_msgs) == 1


@pytest.mark.asyncio
async def test_none_voice_settings_never_sent():
    # All voice keys None ⇒ build_elevenlabs_voice_settings returns None.
    service = _make_service()
    assert service._voice_settings is None
    ws = _FakeWebsocket()
    service._websocket = ws
    service._voice_settings_sent = False

    await _drive_run_tts(service, "hello", "ctx-1")
    await _drive_run_tts(service, "world", "ctx-2")

    assert _voice_settings_messages(ws) == []


@pytest.mark.asyncio
async def test_connect_resets_voice_settings_guard():
    """A fresh socket legally re-sends voice_settings as its first message."""
    service = _make_service(stability=0.8)
    ws1 = _FakeWebsocket()
    service._websocket = ws1
    service._voice_settings_sent = False

    await _drive_run_tts(service, "hello", "ctx-1")
    assert len(_voice_settings_messages(ws1)) == 1

    # Simulate a reconnect: new socket, guard reset (as _connect_websocket does).
    ws2 = _FakeWebsocket()
    service._websocket = ws2
    service._voice_settings_sent = False

    await _drive_run_tts(service, "again", "ctx-2")
    assert len(_voice_settings_messages(ws2)) == 1


@pytest.mark.asyncio
async def test_voice_settings_change_forces_reconnect(monkeypatch):
    """A mid-call voice-settings delta reconnects instead of an in-socket resend."""
    service = _make_service(stability=0.8)
    ws = _FakeWebsocket()
    service._websocket = ws
    service._voice_settings_sent = False

    # First context opens the socket and sends settings.
    await _drive_run_tts(service, "hello", "ctx-1")
    assert service._voice_settings_sent is True

    disconnect = AsyncMock()
    connect = AsyncMock()
    monkeypatch.setattr(service, "_disconnect", disconnect)
    monkeypatch.setattr(service, "_connect", connect)
    # No-op the context-close path so we can detect if it is (wrongly) used.
    close_ctx = AsyncMock()
    monkeypatch.setattr(service, "_close_context", close_ctx)

    await service._update_settings(ElevenLabsTTSSettings(stability=0.5))

    disconnect.assert_awaited()
    connect.assert_awaited()


@pytest.mark.asyncio
async def test_first_context_init_unchanged():
    """Regression: the first context-init message is byte-identical to today's."""
    service = _make_service(stability=0.8, similarity_boost=0.75, speed=1.0)
    ws = _FakeWebsocket()
    service._websocket = ws
    service._voice_settings_sent = False

    await _drive_run_tts(service, "hello", "ctx-1")

    init_msg = next(m for m in ws.sent if m.get("text") == " ")
    assert init_msg["context_id"] == "ctx-1"
    assert init_msg["voice_settings"] == {
        "stability": 0.8,
        "similarity_boost": 0.75,
        "speed": 1.0,
    }
