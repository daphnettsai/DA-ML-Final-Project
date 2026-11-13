# Manhattan Restaurant Alternatives

Enter a Manhattan restaurant you like, confirm the right match, set filters, and click
**Find alternatives** to get comparable restaurants with data-based explanations.
Plain Python and template explanations; no LLM calls at runtime.

> ⚠️ **Yelp terms are UNVERIFIED.** We have not confirmed that Yelp's terms allow our
> analysis, deployment, or display of Yelp data. Owner and date for resolving this: **TBD**.
> The Yelp free trial is time-limited (30 days) and capped at about 5,000 calls.
> Whether to replace Yelp with another licensed source is also **open**.

## Status

Under construction. See [docs/SPEC.md](docs/SPEC.md) for the build spec and approved amendments.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your Yelp key, or leave it empty for local-only mode
```

## Testing

```bash
pytest
```

Tests use synthetic data and mocked Yelp responses only.

## Yelp data rules (summary)

- Yelp content (name, rating, review count, price, categories, coordinates) is cached for
  at most 24 hours. Only Yelp business IDs are stored permanently.
- Yelp is called only when a user clicks **Find alternatives**; if the cache is older than
  24 hours, that click refreshes the fixed restaurant sample first.
- No real Yelp data goes into Git, tests, or fixtures.

_Sections still to write (Phase 8): refresh, methodology, limitations, deployment._
