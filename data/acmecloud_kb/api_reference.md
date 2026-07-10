# API Reference

## Basics

The REST API base URL is https://api.acmecloud.example/v3. The API accepts and
returns JSON only. Authentication uses an AKT passed in the `Authorization`
header.

## Rate limiting

Soft rate limits are per plan: 600 requests per minute on Starter, 3,000 on
Growth, and 20,000 on Enterprise. Short bursts of up to 2× the plan limit are
tolerated for at most 60 seconds. When a limit is exceeded the API returns
HTTP 429; clients should retry with exponential backoff.

## Pagination

List endpoints use cursor-based pagination. The default page size is 50 items
and the maximum `page_size` is 200.

## Idempotency

Mutating requests can pass an `Idempotency-Key` header. Idempotency keys are
valid for 24 hours after first use.

## Webhooks

Webhooks support a maximum payload of 256 KB. Delivery is attempted 5 times
with exponential backoff, and each delivery attempt times out after
10 seconds.

## Error codes

HTTP 429 means the rate limit was exceeded. The application error code
ACME-4031 means the AKT used for the request has expired and must be rotated.
