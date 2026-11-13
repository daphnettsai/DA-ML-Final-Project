# Build spec: decisions and amendments

The full build spec lives with the team. This file records decisions made during the build
that change or clarify it.

## Amendment 1: refresh on click (approved 2026-11-13)

Problem: similarity (Yelp categories) and distance (Yelp coordinates) need Yelp content,
which may be kept at most 24 hours, so Stage 1 cannot rely on a permanent local copy.

Decision:
- **Section 3:** when a user clicks **Find alternatives** and the Yelp cache is missing or
  older than 24 hours, that click refreshes the fixed restaurant sample before ranking.
  Clicks within 24 hours use the cache and make no Yelp calls. Still no scheduled calls
  and no calls on keystrokes or reruns.
- **Section 7 (`refresh.py`):** refresh by rerunning the same Yelp Business Search queries
  used to build the sample (about 30 calls for ~200 restaurants) and keep only business
  IDs already in the sample. A sampled restaurant missing from the new results is labelled
  "not refreshed"; it is never replaced, and its stale data is never used.
- Fallback is unchanged: if Yelp is unavailable, rank from local NYC data only and label
  Yelp fields "unavailable".

## Clarification: ranking rule

The tiered ranking (cuisine band, distance band, price gap, rating, distance, name)
replaces the earlier "score rounded to 2 decimals" rule. Similarity is rounded to
2 decimals before computing the cuisine band, so 0.80 is never placed in the 0.7 band
by floating-point error.

## Still open

- Whether Yelp permits our academic/app use (owner and date TBD).
- Whether to replace Yelp with another licensed source.
- Whether notebooks are committed (outputs stripped) or left out.
- First commit goes directly to `main`, or via another route.
