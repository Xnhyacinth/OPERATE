# Security policy

## Supported versions

The public GitHub repository and public Hugging Face dataset each expose one
current OPERATE snapshot. Report vulnerabilities against that current tree.

## Reporting a vulnerability

Do not open a public issue for security reports that include secrets, credentials,
or exploit details.

Email the maintainer listed in `CITATION.cff` or use GitHub's private vulnerability
reporting for `Xnhyacinth/OPERATE`.

Please include:

- the affected file or command
- a reproduction that does not require leaking credentials
- the impact if the issue is confirmed

## Secrets

Public evaluation uses `API_KEY` and `BASE_URL` only. Never commit tokens,
provider keys, or local `.env` files.
