# Security Policy

## Supported versions

ReflexRoute is currently pre-1.0. Security fixes are applied to the latest
release and the `main` branch.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting flow under the repository's
**Security** tab. Include the affected version, reproduction steps, impact, and
any suggested mitigation.

Do not disclose a vulnerability in a public issue before maintainers have had a
reasonable opportunity to investigate and release a fix. Do not include real
API keys, private prompts, or routing history in a report.

## Credential handling

ReflexRoute reads `OPENROUTER_API_KEY` from the environment and does not persist
it. Keep `.env` files out of version control and rotate any credential that is
accidentally exposed.
