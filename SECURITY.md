# Security Policy

## Supported Versions

Currently, only the `master` branch is supported for security updates. SENTINEL is pre-1.0 software.

| Version | Supported          |
| ------- | ------------------ |
| `master`| :white_check_mark: |
| < 1.0   | :x:                |

## Threat Model

SENTINEL is an offline, self-contained simulator and analytical pipeline. It is **not** currently designed to be exposed to the public internet or untrusted adversarial network traffic (other than the simulated agents inside the engine).

### In Scope
- Vulnerabilities that allow arbitrary code execution when processing malicious data sets (e.g., memory unsafety in the LOB core triggered by malformed order structs).
- Vulnerabilities in the Prometheus metrics exporter (`tiny_http`) that could allow remote code execution or denial of service within a local network deployment.

### Out of Scope
- Denial of Service (DoS) resulting from feeding massive event volumes to the LOB (this is expected behaviour in a simulator).
- Vulnerabilities in the Python dependencies (`scipy`, `pandas`) unless they uniquely exploit SENTINEL's usage pattern.
- Vulnerabilities in the Grafana/Prometheus Docker Compose stack (these are external upstream projects used solely for local visualization).

## Reporting a Vulnerability

If you discover a security vulnerability within SENTINEL, please do not disclose it publicly. 

Instead, open a GitHub Issue with the label `security` or contact the maintainer directly if contact details are provided on their GitHub profile. We aim to acknowledge all reports within 72 hours.
