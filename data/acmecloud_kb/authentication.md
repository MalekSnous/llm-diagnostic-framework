# Authentication and Access

## Tokens

AcmeCloud uses a token called an "AKT" (Acme Key Token). AKTs expire after
14 days by default; the maximum configurable AKT lifetime is 60 days. An AKT can
be rotated without downtime using the `acme auth rotate` command.

Service accounts use a longer-lived token called an "AKT-S", which expires after
90 days. User tokens are prefixed `akt_` and service-account tokens are prefixed
`akts_`, so log scanners can tell them apart.

## Single sign-on

SSO via SAML 2.0 is available on the Growth plan and above. SSO via OIDC is
available on the Enterprise plan only. SCIM user provisioning is available on
the Enterprise plan only and was introduced in release v3.1.

## Account security

Multi-factor authentication (TOTP) has been mandatory for all admin accounts
since release v3.1. After 10 consecutive failed login attempts, an account is
locked for 15 minutes.
