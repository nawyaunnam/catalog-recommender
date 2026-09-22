# Engineering notes: CatalogRecommender — implicit-feedback ranking

## Problem and flow

Timestamped interactions → per-user latest-event holdout → training-only user/item sets → co-occurrence cosine similarity → unseen-item ranking. Every result includes its strongest contributing item; cold-start users receive popularity-ranked items. Evaluation compares hit rate and MRR with popularity on the same eligible users.

## Current boundaries

Binary interactions and in-memory pair counting; expensive for users with very long histories. The tiny synthetic demo is not a business-performance benchmark. No embeddings, online learning, or user identity data are included.

## Interview walkthrough

1. Run the demo and explain each output in terms of the code.
2. Show a test that exercises a failure rather than only a successful call.
3. Trace one input through the core implementation and its stored state.
4. Explain the tradeoff made by the current storage or algorithm choice.
5. Describe what would change with 100× the data or concurrent users.
6. Make a small extension and add a regression test before using this in a resume.

## Validation

See `test_engine.py` for executable assertions and `docs/demo-output.txt` for
captured results. CI is configured but remote CI results are not assumed.
