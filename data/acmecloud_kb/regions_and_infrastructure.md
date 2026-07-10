# Regions and Infrastructure

## Available regions

AcmeCloud is available in three regions: eu-paris-1, us-reston-2, and
ap-osaka-3. The ap-osaka-3 region launched in October 2024.

The default region for new accounts created after January 2025 is eu-paris-1.
Before January 2025, the default region for new accounts was us-reston-2.

## Replication and residency

Cross-region replication is only supported between eu-paris-1 and us-reston-2.
Replication involving ap-osaka-3 is planned but not yet available.

Data residency: data stored by EU customers in eu-paris-1 never leaves that
region. Each region spans at least 3 availability zones, and the inter-region
latency target between replicated regions is under 120 ms.
