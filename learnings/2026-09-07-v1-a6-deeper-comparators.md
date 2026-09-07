# 2026-09-07: Deeper comparator resource preflight

## Context

Configuration 00 passed for three comparator methods. That does not qualify the
frozen deeper-tree or 1000-round search configurations.

## Decision or Result

Add only configuration 05 to the existing bounded comparator selector. It keeps
fold zero, learning rate 0.03, zero regularization, 255 bins and patience 50, with
1000-round maxima, depth 6 for XGBoost/CatBoost and 31 leaves for LightGBM. Reject
other indices and prevent this option from silently dispatching OpenBoost jobs.

## Changes

- a6_resource_preflight accepts --comparators --comparator-config 5.
- Exact frozen configuration selection, resource limits, fresh replay and failure
  retention are reused; the input packet and all method families stay unchanged.

## Verification

Thirty-three focused checks pass across comparator, paired and profiling contracts.
Full regression and remote execution results are recorded below.

## Failed Attempts

None in preparation. Retain any real execution failure without retry or tuning.

## Risks and Follow-ups

The 1000-round budget is a maximum with frozen early stopping; it does not force
1000 completed rounds. Only three jobs are launched and stop on the first failure.
This does not qualify other folds/configurations, deeper OpenBoost resource use,
selected quality, full search or any GPU/authoring/adoption gate. Existing upload
scope is unchanged: approved train/validation packet and allowlisted public source.

## Commits

Commit preparation before remote execution; evidence follows separately. No push.

Preparation validation: 1125 CPU tests passed, one Linux-only skip; 33 focused
checks, lint and docs pass. Real configuration-05 execution remains pending.
