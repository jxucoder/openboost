# 2026-09-06: Distinguish CPU implementation from phase acceptance

## Context

Parent a2a03e4. Current CPU adapters now span A1–A12, while repeated continuation
risked conflating validation plumbing with formal CPU readiness and GPU progress.

## Decision or Result

[This review](../v1-sprints/062-cpu-exit-and-gpu-entry.md) reconciles current
capabilities with unfinished source, workflow, installed D5 and formal E5 gates.
Keep current F3 ordering explicit; no GPU implementation exists. Name the first
B12 vertical path and acceptance without declaring an unapproved phase overlap.

## Changes

- Current capability/application position and a finite ordered remaining list.
- Next slice: installed same-seed/different-ID RNG and changed-data preparation
  probes, complementing existing scheduling permutations/retries/failures.
- GPU first-slice scope and parity/cost criteria; no new CPU objective expansion.

## Verification

Read active F1/F2/F3 and E0/E1/E2/E5 criteria, device construction design,
current worker/search call paths, installed scheduler verifier and sprint evidence.
No new test run claimed; latest regression is Sprint 061's 923 passes. Strict
MkDocs and whitespace checks passed. Sealed held-out tasks were not inspected.

## Failed Attempts

No experimental attempt in this review. Historical missing-objective/stopping
claims were superseded using current evidence rather than copied forward.

## Risks and Follow-ups

No full CPU gate is declared. MSLR/source obligations, real searches, joint A9
selection, current evidence reconciliation and formal author cohorts remain.
Starting GPU early requires an explicit sequencing amendment, not silently
forgetting these obligations. No push or publication.

## Commits

- This review slice; parent a2a03e4.
