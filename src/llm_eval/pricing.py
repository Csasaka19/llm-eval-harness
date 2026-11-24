"""Token pricing tables, loaded from user-supplied TOML.

The harness ships no prices: they change often and differ by provider, region, and
contract. Each table must carry an ``as_of`` date and a ``source`` so reports can say
exactly which prices were used.

Example::

    as_of = "2026-11-01"
    source = "https://example.com/pricing"
    currency = "USD"

    [models."my-model"]
    input_per_mtok = 1.00    # price per 1M input tokens
    output_per_mtok = 4.00   # price per 1M output tokens
"""

from __future__ import annotations

import datetime as dt
import tomllib
from dataclasses import dataclass
from pathlib import Path

from llm_eval.types import Usage


class PricingError(ValueError):
    """Raised for malformed pricing tables."""


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float


@dataclass(frozen=True)
class PriceTable:
    as_of: str
    source: str
    currency: str
    models: dict[str, ModelPrice]

    def cost(self, model: str | None, usage: Usage | None) -> float | None:
        """Cost of one request, or ``None`` if the model or usage is unknown."""
        if model is None or usage is None:
            return None
        price = self.models.get(model)
        if price is None:
            return None
        return (
            usage.input_tokens * price.input_per_mtok + usage.output_tokens * price.output_per_mtok
        ) / 1_000_000


def load_pricing(path: str | Path) -> PriceTable:
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise PricingError(f"{path}: invalid TOML: {exc}") from exc

    as_of = data.get("as_of")
    if isinstance(as_of, dt.date):
        as_of = as_of.isoformat()
    source = data.get("source")
    currency = data.get("currency", "USD")
    if not isinstance(as_of, str) or not as_of:
        raise PricingError(f"{path}: 'as_of' date is required")
    if not isinstance(source, str) or not source:
        raise PricingError(f"{path}: 'source' is required (where the prices came from)")
    if not isinstance(currency, str):
        raise PricingError(f"{path}: 'currency' must be a string")

    raw_models = data.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise PricingError(f'{path}: at least one [models."<id>"] table is required')
    models: dict[str, ModelPrice] = {}
    for model_id, entry in raw_models.items():
        if not isinstance(entry, dict):
            raise PricingError(f"{path}: models.{model_id} must be a table")
        try:
            models[model_id] = ModelPrice(
                input_per_mtok=_price(entry, "input_per_mtok"),
                output_per_mtok=_price(entry, "output_per_mtok"),
            )
        except ValueError as exc:
            raise PricingError(f"{path}: models.{model_id}: {exc}") from exc
    return PriceTable(as_of=as_of, source=source, currency=currency, models=models)


def _price(entry: dict[str, object], key: str) -> float:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key!r} must be a number")
    if value < 0:
        raise ValueError(f"{key!r} must be >= 0")
    return float(value)
