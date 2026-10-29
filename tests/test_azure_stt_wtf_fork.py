#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WTF fork behaviour of AzureSTTService on Pipecat 1.12.

- 1e8b751b4: PostRefinement post-processing (350 ms segmentation lives in test_azure_stt.py).
- a7e1234c1: a >=16 kHz push stream is declared and 8 kHz telephony audio is upsampled
  (Azure closes sub-16 kHz streams with error 1007 since 2026-10-23).
- d463d14ce: two or more candidate locales build a continuous-LID recognizer and the
  detected language rides the final transcript.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pipecat.services.azure.stt as azure_stt
from pipecat.services.azure.stt import AzureSTTService

MOD = "pipecat.services.azure.stt"


def _service(pipeline_rate: int, **kwargs) -> AzureSTTService:
    svc = AzureSTTService(api_key="fake", region="centralindia", **kwargs)
    # STTService.setup() sets this from FrameProcessorSetup.audio_in_sample_rate.
    svc._sample_rate = pipeline_rate
    return svc


class _SdkPatches:
    """Patch the SDK objects _connect builds so no network is touched."""

    def __enter__(self):
        self.fmt = patch(f"{MOD}.AudioStreamFormat").start()
        self.stream = patch(f"{MOD}.PushAudioInputStream").start()
        patch(f"{MOD}.AudioConfig").start()
        self.recognizer = patch(f"{MOD}.SpeechRecognizer").start()
        return self

    def __exit__(self, *exc):
        patch.stopall()


class TestPostProcessing(unittest.TestCase):
    def test_post_refinement_is_set(self):
        from azure.cognitiveservices.speech import PropertyId

        with patch(f"{MOD}.SpeechConfig.set_property") as mock_set:
            AzureSTTService(api_key="fake", region="eastus")
        calls = [
            c.args
            for c in mock_set.call_args_list
            if c.args[0] == PropertyId.SpeechServiceResponse_PostProcessingOption
        ]
        self.assertEqual(calls, [(PropertyId.SpeechServiceResponse_PostProcessingOption, "PostRefinement")])


class TestSixteenKilohertzStream(unittest.IsolatedAsyncioTestCase):
    def test_declared_rate_never_below_16k(self):
        rate_for = AzureSTTService.azure_stream_sample_rate_for
        self.assertEqual(rate_for(8000), 16000)
        self.assertEqual(rate_for(16000), 16000)
        self.assertEqual(rate_for(24000), 24000)
        self.assertEqual(rate_for(0), 16000)
        self.assertEqual(azure_stt.AZURE_MIN_STREAM_SAMPLE_RATE, 16000)

    async def test_8k_declares_16k_and_upsamples_every_chunk(self):
        with _SdkPatches() as sdk:
            svc = _service(8000)
            await svc._connect()
            self.assertEqual(sdk.fmt.call_args.kwargs["samples_per_second"], 16000)

            written: list[bytes] = []
            svc._audio_stream.write = written.append
            chunk_8k = bytes(range(256)) + bytes(64)  # 20 ms at 8 kHz = 320 bytes
            for _ in range(10):
                async for _ in svc.run_stt(chunk_8k):
                    pass
        self.assertEqual(len(written), 10, "every 20 ms chunk must produce output")
        self.assertTrue(all(len(w) > 0 and len(w) % 2 == 0 for w in written))
        total = sum(len(w) for w in written)
        self.assertTrue(0.97 * 6400 <= total <= 6400, total)

    async def test_16k_passes_through_byte_for_byte(self):
        with _SdkPatches() as sdk:
            svc = _service(16000)
            await svc._connect()
            self.assertEqual(sdk.fmt.call_args.kwargs["samples_per_second"], 16000)
            written: list[bytes] = []
            svc._audio_stream.write = written.append
            pcm = bytes(range(256)) * 4
            async for _ in svc.run_stt(pcm):
                pass
        self.assertEqual(written, [pcm])
        self.assertIsNone(svc._upsampler)

    async def test_empty_audio_is_never_written(self):
        with _SdkPatches():
            svc = _service(8000)
            await svc._connect()
            svc._audio_stream.write = MagicMock()
            async for _ in svc.run_stt(b""):
                pass
        svc._audio_stream.write.assert_not_called()

    async def test_reconnect_resets_the_upsampler(self):
        """A settings-driven reconnect must not carry filter state into the new stream."""
        with _SdkPatches():
            svc = _service(8000)
            await svc._connect()
            svc._audio_stream.write = MagicMock()
            async for _ in svc.run_stt(bytes(320)):
                pass
            self.assertIsNotNone(svc._upsampler)
            await svc._disconnect()
            self.assertIsNone(svc._upsampler)
            await svc._connect()
            self.assertIsNone(svc._upsampler)


class TestContinuousLanguageIdentification(unittest.IsolatedAsyncioTestCase):
    async def test_two_candidates_build_continuous_lid_recognizer(self):
        from azure.cognitiveservices.speech import PropertyId

        with _SdkPatches() as sdk, patch(f"{MOD}.SpeechConfig.set_property") as mock_set:
            svc = _service(8000, candidate_languages=["hi-IN", " en-IN ", ""])
            await svc._connect()
        kwargs = sdk.recognizer.call_args.kwargs
        self.assertIn("auto_detect_source_language_config", kwargs)
        self.assertEqual(svc._candidate_languages, ["hi-IN", "en-IN"])
        mock_set.assert_any_call(PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous")

    async def test_one_candidate_keeps_single_language_recognizer(self):
        with _SdkPatches() as sdk:
            svc = _service(8000, candidate_languages=["hi-IN"])
            await svc._connect()
        self.assertNotIn("auto_detect_source_language_config", sdk.recognizer.call_args.kwargs)
        self.assertEqual(svc._candidate_languages, [])

    def test_detected_language_used_only_when_lid_is_on(self):
        lid = _service(8000, candidate_languages=["hi-IN", "en-IN"])
        single = _service(8000)
        result = SimpleNamespace()
        with patch(f"{MOD}.AutoDetectSourceLanguageResult", return_value=SimpleNamespace(language="en-IN")):
            self.assertEqual(lid._detected_language(result), "en-IN")
            self.assertIsNone(single._detected_language(result))
        with patch(f"{MOD}.AutoDetectSourceLanguageResult", return_value=SimpleNamespace(language="")):
            self.assertIsNone(lid._detected_language(result))


if __name__ == "__main__":
    unittest.main()
