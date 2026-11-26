"""Scripted stand-in for `anthropic.Anthropic` (no network): returns queued responses, records every request."""

import copy
from types import SimpleNamespace


class _Block(SimpleNamespace):
    def model_dump(self):
        return dict(vars(self))


def tool_use(id, name, input):
    return _Block(type="tool_use", id=id, name=name, input=input)


def text(t):
    return _Block(type="text", text=t)


def response(blocks, stop_reason, input_tokens, output_tokens):
    r = SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )
    r.model_dump = lambda: {
        "content": [b.model_dump() for b in blocks],
        "stop_reason": stop_reason,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
    return r


class FakeAnthropic:
    def __init__(self, responses: list):
        self.responses, self.calls = list(responses), []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(copy.deepcopy(kw))
        return self.responses.pop(0)
