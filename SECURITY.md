# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, email **security@scout-scraper.dev** with:

- A description of the vulnerability
- Steps to reproduce the issue
- The potential impact
- Any suggested fix (optional)

### Response Timeline

- **Acknowledgment**: within 48 hours
- **Status update**: within 7 days
- **Fix or mitigation**: as soon as reasonably possible, depending on severity

## Scope

The following are considered in-scope for security reports:

- **Sandbox escapes** -- bypasses of the RestrictedPython sandbox that isolate AI-generated code
- **Code injection** -- vulnerabilities in generated scraping scripts that could lead to arbitrary code execution
- **Dependency vulnerabilities** -- security issues in third-party packages used by Scout

## Architecture Note

Scout executes AI-generated code in a **RestrictedPython sandbox** to prevent untrusted code from accessing the filesystem, network, or operating system outside of intended operations. Reports of sandbox bypass techniques are treated with the highest priority.

## Disclosure

We follow coordinated disclosure. We ask that you give us reasonable time to address the issue before making any information public.
