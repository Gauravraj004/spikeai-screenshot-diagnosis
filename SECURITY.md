# Security Policy

Report suspected vulnerabilities privately to **gauravraj93689@gmail.com**.

## Sensitive inputs

Screenshots and HTML captures may contain credentials, personal data, session identifiers, or confidential customer information. Redact sensitive content before sending it to an external model provider.

## Credentials

Keep `.env` local and never commit API keys. Use `.env.example` as the template.

## Model-provider considerations

Review the configured provider's retention, privacy, and enterprise-data settings before processing production captures. Model-generated diagnoses are advisory and should not be treated as a security verdict without independent verification.
