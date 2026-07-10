# AcmeCloud Platform Overview

AcmeCloud is Acme Corp's internal data-orchestration platform, launched in 2021.
This knowledge base is the source of truth for support agents. Every fact in it is
specific to AcmeCloud and is NOT public knowledge — a general-purpose LLM cannot
know these details without retrieval.

## Components

The platform is built around four core components. **Pipelines** are managed
compute jobs that move and transform data. **Datasets** are versioned, managed
storage objects. **Flows** are orchestration DAGs that chain pipelines together;
Flows were introduced in release v3.0. **Lenses** are the built-in analytics
dashboards; Lenses became generally available in release v3.2.

## Current release

The current platform release is v3.2, codenamed "Meridian", released in
January 2026. AcmeCloud ships two official interfaces: the web console and the
`acme` command-line tool. The platform itself runs on Kubernetes clusters
operated by Acme Corp.

## Service status

The public service-status page is status.acmecloud.example. Incidents and
scheduled maintenance windows are announced there at least 72 hours in advance.
