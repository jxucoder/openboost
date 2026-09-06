# 2026-09-06: Housing hosted license evidence

## Context

The original data freeze recorded no archive license and inaccessible StatLib
source pages. Public availability alone did not establish a license.

## Decision or Result

Figshare article 3829992 version 2 declares CC BY 4.0 on file 5976036.
Its published MD5 matches the downloaded archive, whose SHA256 also matches
the original Housing freeze. Record this dated host declaration without rewriting
historical data/preprocessing evidence or claiming original-source rights review.

## Changes

- [Review](../benchmarks/v1/datasets/housing-license-review.json) retains public
  API metadata, response digest, omitted account URL field names and file hashes.
- Evaluation README links the declaration and includes uploader attribution.

## Verification

Fetched public `https://api.figshare.com/v2/articles/3829992`. Independently
computed archive MD5 and SHA256 match the host and existing data freeze.
No model outputs were examined and no split or raw data was changed.

## Failed Attempts

Microsoft's linked MSLR agreement through the public OneDrive shares endpoint
returned HTTP 401. No authenticated access was bypassed. A4 source preparation
remains open; Veteran original-source license evidence also remains open.

## Risks and Follow-ups

The Figshare declaration is the uploader's hosted-file declaration. It is not
a separate verification of the original StatLib rights chain. Finish the remaining
source records before declaring the dataset gate complete.

## Commits

- This slice: `docs: record matching Housing host license evidence`.
