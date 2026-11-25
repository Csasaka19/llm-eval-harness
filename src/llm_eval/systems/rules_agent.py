"""Baseline agent: regular-expression rules that map requests to tool calls.

It handles the phrasings it was written for and nothing else, which is the point: it
shows what "no understanding" scores, and it fails visibly on paraphrases. Calls are
emitted in the order their triggers appear in the request.
"""

from __future__ import annotations

import re

from llm_eval.types import EvalCase, JSONObject, SystemResponse, ToolCall

_WEATHER = re.compile(r"weather in ([A-Z][a-z]+)(?:\s+and\s+(?:in\s+)?([A-Z][a-z]+))?")
_CONVERT = re.compile(r"convert\s+([\d.]+)\s+([A-Za-z]{3})\s+to\s+([A-Za-z]{3})", re.I)
_TICKET_TITLED = re.compile(r"ticket titled '([^']+)'", re.I)
_TICKET_COLON = re.compile(r"ticket:\s*(.+?)\s*$", re.I)
_PRIORITY = re.compile(r"\b(low|medium|high)[- ]priority\b", re.I)
_FLIGHT_LOG = re.compile(r"flight log for (?:drone )?([A-Z]{2}-\d+)")
_SEARCH = re.compile(r"search the docs for (?:the )?(.+?)(?:,|\s+then\b|\.|$)", re.I)


class RulesAgent:
    name = "rules-agent"

    def run(self, case: EvalCase) -> SystemResponse:
        text = case.input
        found: list[tuple[int, ToolCall]] = []

        if m := _WEATHER.search(text):
            unit = "fahrenheit" if "fahrenheit" in text.lower() else "celsius"
            for group in (1, 2):
                if city := m.group(group):
                    found.append((m.start(group), _call("get_weather", city=city, unit=unit)))

        for m in _CONVERT.finditer(text):
            amount = float(m.group(1))
            found.append(
                (
                    m.start(),
                    _call(
                        "convert_currency",
                        amount=int(amount) if amount.is_integer() else amount,
                        **{"from": m.group(2).upper(), "to": m.group(3).upper()},
                    ),
                )
            )

        if m := _TICKET_TITLED.search(text) or _TICKET_COLON.search(text):
            p = _PRIORITY.search(text)
            priority = p.group(1).lower() if p else "medium"
            found.append((m.start(), _call("create_ticket", title=m.group(1), priority=priority)))

        if m := _FLIGHT_LOG.search(text):
            found.append((m.start(), _call("get_flight_log", drone_id=m.group(1))))

        if m := _SEARCH.search(text):
            found.append((m.start(), _call("search_docs", query=m.group(1))))

        calls = tuple(call for _, call in sorted(found, key=lambda pair: pair[0]))
        answer = (
            "Calling: " + ", ".join(c.name for c in calls) if calls else "I can't help with that."
        )
        return SystemResponse(answer=answer, tool_calls=calls, model="rules")


def _call(name: str, **arguments: str | int | float) -> ToolCall:
    args: JSONObject = dict(arguments)
    return ToolCall(name=name, arguments=args)
