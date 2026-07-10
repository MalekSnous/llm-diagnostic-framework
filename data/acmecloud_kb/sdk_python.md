# Python SDK

## Package

The official Python SDK is the `acmecloud-sdk` package, installed with
`pip install acmecloud-sdk`. It requires Python 3.10 or newer.

The legacy `acme-py` package is deprecated and will be removed in v4.0,
scheduled for March 2026. New integrations must use `acmecloud-sdk`.

## Client behaviour

The SDK client retries failed requests 3 times by default. The default request
timeout is 30 seconds. Async support (`AcmeAsyncClient`) has been available
since SDK version 2.5.

The client region is picked in this order: the `region` argument to
`AcmeClient(...)`, then the `ACME_REGION` environment variable, then the
account default region.
