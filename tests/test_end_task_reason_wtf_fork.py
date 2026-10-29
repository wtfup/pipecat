#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WTF fork: EndTaskReason keeps every disposition value WTF Voice stores.

Dograh dropped several members in 2c00dc24d. WTF Voice persists these values on
workflow runs and keys reports/UI on them, so removing or renaming one breaks
call endings at runtime (AttributeError in the end-call tool) and splits
historical data.
"""

from pipecat.utils.enums import EndTaskReason

# Every value the WTF Voice app emits or stores. Values are a data contract.
REQUIRED = {
    "CALL_DURATION_EXCEEDED": "call_duration_exceeded",
    "CALL_TRANSFERRED": "call_transferred",
    "END_CALL_TOOL_REASON": "end_call_tool",
    "VOICEMAIL_DETECTED": "voicemail_detected",
    "USER_IDLE_MAX_DURATION_EXCEEDED": "user_idle_max_duration_exceeded",
    "USER_HANGUP": "user_hangup",
    "USER_QUALIFIED": "user_qualified",
    "USER_DISQUALIFIED": "user_disqualified",
    "SYSTEM_CANCELLED": "system_cancelled",
    "SYSTEM_CONNECT_ERROR": "system_connect_error",
    "UNEXPECTED_ERROR": "unexpected_error",
    "UNKNOWN": "unknown",
    "TRANSFER_CALL": "transfer_call",
    "PIPELINE_ERROR": "pipeline_error",
}


def test_wtf_voice_dispositions_are_present_with_stable_values():
    for name, value in REQUIRED.items():
        assert EndTaskReason[name].value == value, name


def test_values_are_unique():
    values = [m.value for m in EndTaskReason]
    assert len(values) == len(set(values))
