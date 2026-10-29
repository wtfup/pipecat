#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure Speech-to-Text service implementation for Pipecat.

This module provides speech-to-text functionality using Azure Cognitive Services
Speech SDK for real-time audio transcription.
"""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import soxr
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.azure.common import language_to_azure_language
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import AZURE_TTFS_P99
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

try:
    from azure.cognitiveservices.speech import (
        CancellationReason,
        ProfanityOption,
        PropertyId,
        ResultReason,
        SpeechConfig,
        SpeechRecognizer,
    )

    try:  # continuous language identification (optional on older SDKs)
        from azure.cognitiveservices.speech import (
            AutoDetectSourceLanguageConfig,
            AutoDetectSourceLanguageResult,
        )
    except ImportError:  # pragma: no cover - SDK drift guard
        AutoDetectSourceLanguageConfig = None  # type: ignore[assignment,misc]
        AutoDetectSourceLanguageResult = None  # type: ignore[assignment,misc]
    from azure.cognitiveservices.speech.audio import (
        AudioStreamFormat,
        PushAudioInputStream,
    )
    from azure.cognitiveservices.speech.dialog import AudioConfig
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Azure, you need to `uv add "pipecat-ai[azure]"`.')
    raise ImportError(f"Missing module: {e}") from e


# WTF fork: fast end-of-utterance segmentation. Azure's default trailing-silence
# window is ~500 ms; 350 ms shaves that off the STT finalization leg of every
# turn (army audit 2026-09-13, top-5 fix list). Upstream 1.12 exposes this as the
# ``segmentation_silence_timeout_ms`` setting with default ``None``; the fork
# defaults it to 350 so callers keep today's latency. Pass ``None`` explicitly in
# settings to fall back to Azure's default.
AZURE_SEGMENTATION_SILENCE_TIMEOUT_MS = 350
# WTF fork: post-processing = PostRefinement rewrites only the FINAL transcript
# and leaves interim hypotheses untouched, so interim consumers stay fast.
AZURE_POST_PROCESSING_OPTION = "PostRefinement"
# WTF fork: Azure's streaming recognizer stopped accepting 8 kHz push streams on
# 2026-10-23 (every telephony call died at answer with ``Error code: 1007 ...
# Named streaming recognition requires 16-kHz mono 16-bit PCM``; wtf-voice runs
# 1384-1533). Telephony transports (IVR Solutions, Plivo, Telnyx, ARI) deliver
# 8 kHz, so the service declares a 16 kHz push stream and upsamples whenever the
# pipeline rate is below that. Rates at or above 16 kHz pass through untouched.
AZURE_MIN_STREAM_SAMPLE_RATE = 16000

AzureProfanity = Literal["raw", "masked", "removed"]
"""How Azure handles profanity in transcripts.

* ``"raw"`` — return the text as recognized, no masking.
* ``"masked"`` — replace profane words with ``****`` (Azure default).
* ``"removed"`` — drop profane words from the output.
"""

_PROFANITY_OPTIONS: dict[AzureProfanity, ProfanityOption] = {
    "raw": ProfanityOption.Raw,
    "masked": ProfanityOption.Masked,
    "removed": ProfanityOption.Removed,
}


@dataclass
class AzureSTTSettings(STTSettings):
    """Settings for AzureSTTService.

    ``model`` and ``language`` are inherited from ``STTSettings`` /
    ``ServiceSettings``.

    Parameters:
        profanity: How Azure handles profanity in transcripts. One of
            ``"raw"``, ``"masked"``, or ``"removed"`` (see ``AzureProfanity``).
            Store-mode default is ``None`` (Azure SDK default = ``"masked"``).
            Use ``"raw"`` for non-English deployments where Azure's profanity
            list is over-eager and masks ordinary words (e.g. Italian names
            containing common substrings), which breaks downstream fuzzy
            matching and LLM reasoning. See `SpeechConfig.set_profanity
            <https://learn.microsoft.com/en-us/python/api/azure-cognitiveservices-speech/azure.cognitiveservices.speech.speechconfig#azure-cognitiveservices-speech-speechconfig-set-profanity>`_.
        segmentation_silence_timeout_ms: How much silence Azure allows inside a
            phrase, in milliseconds, before it finalizes the recognition and
            emits a ``TranscriptionFrame``. Azure accepts 100–5000; its default
            is 500. Store-mode default is ``None`` (keep Azure's default).
            Lower values finalize sooner, at the cost of splitting phrases that
            contain pauses; higher values keep slow or hesitant speech in one
            transcript. See `phrase segmentation
            <https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-recognize-speech#change-how-silence-is-handled>`_.
    """

    profanity: AzureProfanity | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    segmentation_silence_timeout_ms: int | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )


class AzureSTTService(STTService):
    """Azure Speech-to-Text service for real-time audio transcription.

    This service uses Azure Cognitive Services Speech SDK to convert speech
    audio into text transcriptions. It supports continuous recognition and
    provides real-time transcription results with timing information.
    """

    Settings = AzureSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        region: str | None = None,
        language: Language | None = Language.EN_US,
        sample_rate: int | None = None,
        private_endpoint: str | None = None,
        endpoint_id: str | None = None,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = AZURE_TTFS_P99,
        candidate_languages: list[str] | None = None,
        **kwargs,
    ):
        """Initialize the Azure STT service.

        Args:
            api_key: Azure Cognitive Services subscription key.
            region: Azure region for the Speech service (e.g., 'eastus').
                Required unless ``private_endpoint`` is provided.
            language: Language for speech recognition. Defaults to English (US).

                .. deprecated:: 0.0.105
                    Use ``settings=AzureSTTService.Settings(language=...)`` instead.
                    Will be removed in 2.0.0.

            sample_rate: Audio sample rate in Hz. If None, uses service default.
            private_endpoint: Private endpoint for STT behind firewall.
                See https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-services-private-link?tabs=portal
            endpoint_id: Custom model endpoint id.
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            candidate_languages: WTF fork. Two or more BCP-47 locales switch the
                recognizer to Azure continuous language identification; each
                result then carries its detected language. One or none keeps the
                single-locale recognizer.
            **kwargs: Additional arguments passed to parent STTService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model=None,
            language=Language.EN_US,
            profanity=None,
            segmentation_silence_timeout_ms=AZURE_SEGMENTATION_SILENCE_TIMEOUT_MS,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if language is not None and language != Language.EN_US:
            self._warn_init_param_moved_to_settings("language", "language")
            default_settings.language = language

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )

        # WTF fork: the rate the Azure push stream is declared at; fixed in
        # _connect once the pipeline rate is known. Below 16 kHz the input is
        # upsampled through a stateful soxr stream in "QQ" quality: continuous
        # across 20 ms chunks with no start-up latency (the VHQ stream resampler
        # buffers ~100 ms and returns empty chunks, which would delay every
        # caller's first words). Telephony audio is band-limited, so quick cubic
        # interpolation loses nothing the recognizer can use.
        self._azure_stream_sample_rate: int | None = None
        self._upsampler: "soxr.ResampleStream | None" = None

        # WTF fork: continuous language identification (2026-10-28). Indian
        # callers switch between Hindi and English inside one call; a single
        # hi-IN recognizer writes the English in Devanagari, which the LLM then
        # reads as Hindi.
        cleaned = [
            c.strip() for c in (candidate_languages or []) if isinstance(c, str) and c.strip()
        ]
        if len(cleaned) >= 2 and AutoDetectSourceLanguageConfig is None:
            logger.warning(
                "Azure STT: this SDK has no AutoDetectSourceLanguageConfig; continuous LID is off"
            )
            cleaned = []
        self._candidate_languages: list[str] = cleaned if len(cleaned) >= 2 else []

        recognition_language = assert_given(
            default_settings.language
        ) or language_to_azure_language(Language.EN_US)

        if not region and not private_endpoint:
            raise ValueError("Either 'region' or 'private_endpoint' must be provided.")

        if private_endpoint:
            if region:
                logger.warning(
                    "Both 'region' and 'private_endpoint' provided; 'region' will be ignored."
                )
            self._speech_config = SpeechConfig(
                subscription=api_key,
                endpoint=private_endpoint,
                speech_recognition_language=recognition_language,
            )
        else:
            self._speech_config = SpeechConfig(
                subscription=api_key,
                region=region,
                speech_recognition_language=recognition_language,
            )

        if endpoint_id:
            self._speech_config.endpoint_id = endpoint_id

        self._apply_profanity()
        self._apply_segmentation_silence_timeout()
        self._apply_post_processing()

        self._audio_stream = None
        self._speech_recognizer = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate performance metrics.

        Returns:
            True as this service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Azure service-specific language code.

        Args:
            language: The language to convert.

        Returns:
            The Azure-specific language identifier, or None if not supported.
        """
        return language_to_azure_language(language)

    def _apply_profanity(self):
        """Apply the current ``profanity`` setting to the speech config.

        A no-op when profanity is ``None`` (keeps the Azure SDK default of
        ``"masked"``).
        """
        # Annotate the local so pyright solves ``assert_given``'s TypeVar to the
        # literal instead of widening it to ``str`` (which wouldn't be a valid
        # ``_PROFANITY_OPTIONS`` key).
        profanity: AzureProfanity | None = assert_given(self._settings.profanity)
        if profanity is not None:
            self._speech_config.set_profanity(_PROFANITY_OPTIONS[profanity])

    def _apply_segmentation_silence_timeout(self):
        """Apply the current ``segmentation_silence_timeout_ms`` setting.

        A no-op when the setting is ``None`` (keeps Azure's default of 500 ms).
        """
        timeout_ms = assert_given(self._settings.segmentation_silence_timeout_ms)
        if timeout_ms is not None:
            self._speech_config.set_property(
                PropertyId.Speech_SegmentationSilenceTimeoutMs, str(timeout_ms)
            )

    def _apply_post_processing(self):
        """WTF fork: final-only post-refinement (defensive against SDK drift)."""
        try:
            self._speech_config.set_property(
                PropertyId.SpeechServiceResponse_PostProcessingOption,
                AZURE_POST_PROCESSING_OPTION,
            )
        except Exception as exc:  # pragma: no cover - SDK drift guard
            logger.warning(f"Azure STT: could not set post-processing option ({exc})")

    @staticmethod
    def azure_stream_sample_rate_for(pipeline_rate: int) -> int:
        """The rate to declare to Azure for a given pipeline rate (never below 16 kHz)."""
        return max(int(pipeline_rate or 0), AZURE_MIN_STREAM_SAMPLE_RATE)

    def _audio_for_azure(self, audio: bytes) -> bytes:
        """Upsample pipeline audio to the declared Azure stream rate when they differ."""
        target = self._azure_stream_sample_rate
        if target is None or not audio or target == self.sample_rate:
            return audio
        if self._upsampler is None:
            self._upsampler = soxr.ResampleStream(
                in_rate=self.sample_rate,
                out_rate=target,
                num_channels=1,
                dtype="int16",
                quality="QQ",
            )
        samples = np.frombuffer(audio, dtype=np.int16)
        return self._upsampler.resample_chunk(samples).tobytes()

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if a recognizer setting changed."""
        changed = await super()._update_settings(delta)

        if "language" in changed:
            self._speech_config.speech_recognition_language = assert_given(
                self._settings.language
            ) or language_to_azure_language(Language.EN_US)

        if "profanity" in changed:
            self._apply_profanity()

        if "segmentation_silence_timeout_ms" in changed:
            self._apply_segmentation_silence_timeout()

        # These settings are baked into the recognizer at connect time, so a
        # live change only takes effect after a reconnect.
        if (
            changed.keys()
            & {
                "language",
                "profanity",
                "segmentation_silence_timeout_ms",
            }
            and self._audio_stream
        ):
            await self._disconnect()
            await self._connect()

        return changed

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Process audio data for speech-to-text conversion.

        Feeds audio data to the Azure speech recognizer for processing.
        Recognition results are handled asynchronously through callbacks.

        Args:
            audio: Raw audio bytes to process.

        Yields:
            Frame: Either None for successful processing or ErrorFrame on failure.
        """
        try:
            if self._audio_stream:
                data = self._audio_for_azure(audio)
                # The stream resampler holds a few samples of filter history, so
                # a chunk can come back empty; never hand Azure an empty buffer.
                if data:
                    self._audio_stream.write(data)
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def cleanup(self):
        """Release resources at pipeline teardown."""
        await super().cleanup()
        await self._disconnect()

    async def stop(self, frame: EndFrame):
        """Stop the speech recognition service.

        Args:
            frame: Frame indicating the end of processing.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the speech recognition service.

        Args:
            frame: Frame indicating cancellation.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        """Initialize the Azure speech recognizer and begin continuous recognition."""
        if self._audio_stream:
            return

        try:
            self._azure_stream_sample_rate = self.azure_stream_sample_rate_for(self.sample_rate)
            self._upsampler = None
            if self._azure_stream_sample_rate != self.sample_rate:
                logger.info(
                    f"Azure STT: pipeline audio is {self.sample_rate} Hz; declaring a "
                    f"{self._azure_stream_sample_rate} Hz push stream and upsampling "
                    "(Azure rejects sub-16 kHz streaming since 2026-10-23)."
                )
            stream_format = AudioStreamFormat(
                samples_per_second=self._azure_stream_sample_rate, channels=1
            )
            self._audio_stream = PushAudioInputStream(stream_format)

            audio_config = AudioConfig(stream=self._audio_stream)

            if self._candidate_languages:
                try:
                    self._speech_config.set_property(
                        PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous"
                    )
                except Exception as exc:  # pragma: no cover - SDK drift guard
                    logger.warning(f"Azure STT: could not set continuous LID mode ({exc})")
                logger.info(
                    f"Azure STT: continuous language identification over {self._candidate_languages}"
                )
                self._speech_recognizer = SpeechRecognizer(
                    speech_config=self._speech_config,
                    auto_detect_source_language_config=AutoDetectSourceLanguageConfig(
                        languages=list(self._candidate_languages)
                    ),
                    audio_config=audio_config,
                )
            else:
                self._speech_recognizer = SpeechRecognizer(
                    speech_config=self._speech_config, audio_config=audio_config
                )
            self._speech_recognizer.recognizing.connect(self._on_handle_recognizing)
            self._speech_recognizer.recognized.connect(self._on_handle_recognized)
            self._speech_recognizer.canceled.connect(self._on_handle_canceled)
            self._speech_recognizer.start_continuous_recognition_async()
        except Exception as e:
            await self.push_error(
                error_msg=f"Uncaught exception during initialization: {e}", exception=e
            )

    async def _disconnect(self):
        """Stop recognition and close audio streams."""
        if self._speech_recognizer:
            self._speech_recognizer.stop_continuous_recognition_async()
            self._speech_recognizer = None

        if self._audio_stream:
            self._audio_stream.close()
            self._audio_stream = None
        self._upsampler = None

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    def _detected_language(self, result) -> str | None:
        """The language continuous LID attached to a result, or None when LID is off/absent."""
        if not self._candidate_languages or AutoDetectSourceLanguageResult is None:
            return None
        try:
            detected = AutoDetectSourceLanguageResult(result).language
        except Exception:  # pragma: no cover - SDK drift guard
            return None
        return detected or None

    def _on_handle_recognized(self, event):
        if event.result.reason == ResultReason.RecognizedSpeech and len(event.result.text) > 0:
            # Technically either source could be a raw string, but Language is
            # a StrEnum so downstream handles either.
            language = cast(
                "Language | None",
                self._detected_language(event.result)
                or getattr(event.result, "language", None)
                or assert_given(self._settings.language),
            )
            # Azure's ``RecognizedSpeech`` event is by definition the final
            # recognition for an utterance — mark the frame as such so that
            # downstream turn-stop strategies (``SpeechTimeoutUserTurnStop``
            # and friends) can take their finalized fast-path instead of
            # waiting for VAD events that may never arrive on short replies.
            frame = TranscriptionFrame(
                event.result.text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=event,
                finalized=True,
            )
            asyncio.run_coroutine_threadsafe(
                self._handle_transcription(event.result.text, True, language), self.get_event_loop()
            )
            # Report usage before the transcription frame so tracing can attach
            # it to the STT span the frame closes (submissions run in order).
            asyncio.run_coroutine_threadsafe(self.emit_stt_usage_metrics(), self.get_event_loop())
            asyncio.run_coroutine_threadsafe(self.push_frame(frame), self.get_event_loop())

    def _on_handle_recognizing(self, event):
        if event.result.reason == ResultReason.RecognizingSpeech and len(event.result.text) > 0:
            # Technically either source could be a raw string, but Language is
            # a StrEnum so downstream handles either.
            language = cast(
                "Language | None",
                getattr(event.result, "language", None) or assert_given(self._settings.language),
            )
            frame = InterimTranscriptionFrame(
                event.result.text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=event,
            )
            asyncio.run_coroutine_threadsafe(self.push_frame(frame), self.get_event_loop())

    def _on_handle_canceled(self, event):
        details = event.result.cancellation_details
        if details.reason == CancellationReason.Error:
            error_msg = f"Azure STT recognition canceled: {details.reason}"
            if details.error_details:
                error_msg += f" - {details.error_details}"
            asyncio.run_coroutine_threadsafe(
                self.push_error(error_msg=error_msg), self.get_event_loop()
            )
