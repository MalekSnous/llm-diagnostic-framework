# Troubleshooting Guide

## Common errors

Error code ACME-4031 means the AKT has expired: rotate it with
`acme auth rotate`. HTTP 429 means the plan's API rate limit was exceeded;
retry with exponential backoff or upgrade the plan.

A pipeline stuck in the PENDING state usually means the plan's concurrent
pipeline quota is already fully used; the pipeline starts as soon as a slot
frees up.

A webhook that is never delivered is most often over the 256 KB payload limit;
deliveries are retried only 5 times, after which the event is dropped.

## Runtime limits

The maximum pipeline runtime is 24 hours by default. On the Enterprise plan it
can be raised to 72 hours per pipeline.

Dataset uploads are limited to 5 GB per file through the web console. Through
the CLI, multipart upload raises the limit to 50 GB per file.

## Getting help

Check status.acmecloud.example first for ongoing incidents. Attach a
diagnostics bundle generated with `acme doctor --bundle` to support tickets to
speed up triage.
