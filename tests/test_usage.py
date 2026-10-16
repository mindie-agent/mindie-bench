import json

from mindie_bench.runner.usage import UsageMeter, sum_usage


def sse(meter, value):
    payload = b"data: " + json.dumps(value).encode() + b"\n\n"
    meter.feed(payload[:17])
    meter.feed(payload[17:])


def test_openai_cache_metering_across_chunk_boundaries():
    meter = UsageMeter()
    sse(
        meter,
        {
            "type": "response.completed",
            "response": {
                "usage": {
                    "input_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 80},
                    "output_tokens": 10,
                }
            },
        },
    )
    assert meter.tokens.input_uncached == 20
    assert meter.tokens.input_cached == 80
    assert meter.tokens.output == 10
    assert meter.tokens.complete


def test_anthropic_usage_is_cumulative_not_delta_summed():
    meter = UsageMeter()
    sse(
        meter,
        {
            "type": "message_start",
            "message": {
                "usage": {
                    "input_tokens": 20,
                    "cache_read_input_tokens": 30,
                    "cache_creation_input_tokens": 5,
                }
            },
        },
    )
    sse(meter, {"type": "message_delta", "usage": {"output_tokens": 3}, "delta": {}})
    sse(
        meter,
        {"type": "message_delta", "usage": {"output_tokens": 8}, "delta": {"stop_reason": "end_turn"}},
    )
    assert meter.tokens.input_uncached == 25
    assert meter.tokens.cache_creation == 5
    assert meter.tokens.input_cached == 30
    assert meter.tokens.output == 8
    assert meter.tokens.complete


def test_unknown_request_preserves_partial_usage_and_withholds_complete():
    known, interrupted = UsageMeter(), UsageMeter()
    sse(known, {"usage": {"input_tokens": 10, "output_tokens": 5}})
    sse(interrupted, {"type": "message_start", "message": {"usage": {"input_tokens": 20}}})
    tokens = sum_usage({1: known, 2: interrupted}, 2)
    assert tokens.input_uncached == 30
    assert tokens.output == 5
    assert not tokens.complete


def test_invalid_or_missing_usage_is_not_complete():
    meter = UsageMeter()
    sse(
        meter,
        {"type": "message_delta", "usage": {"output_tokens": 5}, "delta": {"stop_reason": "end_turn"}},
    )
    assert not meter.tokens.complete
    sse(meter, {"usage": {"input_tokens": 10, "input_tokens_details": None, "output_tokens": 1}})
    assert not meter.tokens.complete
