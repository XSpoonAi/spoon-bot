---
name: service_expose
description: Run user-created frontend/backend/WebSocket services and expose them through Cloudflare by default when the user asks to start, deploy, preview, share, or make the service accessible. Local-only is allowed only when explicitly requested.
version: 1.0.0
author: XSpoon Team
tags: [service, frontend, backend, cloudflare, tunnel, background]
default_active: true
triggers:
  - type: keyword
    keywords: [background service, run service, expose service, cloudflare tunnel, trycloudflare, frontend preview, backend preview, websocket, public link]
    priority: 90
scripts:
  enabled: true
  definitions:
    - name: service_expose
      description: Start, stop, inspect, log, and Cloudflare-expose local frontend/backend services.
      type: python
      file: scripts/service_expose.py
      timeout: 60
      input_schema:
        type: object
        properties:
          action:
            type: string
            description: "start, tunnel, expose, start_tunnel, status, list, inspect, logs, stop, or stop_tunnel"
          name:
            type: string
            description: "Stable service name"
          command:
            type: string
            description: "Command to run for action=start"
          cwd:
            type: string
            description: "Working directory for the service command"
          port:
            type: integer
            description: "Local port to expose"
          host:
            type: string
            description: "Local host, default 127.0.0.1"
          url:
            type: string
            description: "Full local URL to expose, for example http://127.0.0.1:3000"
          start_tunnel:
            type: boolean
            description: "Start Cloudflare tunnel after service start"
          replace:
            type: boolean
            description: "Stop and replace an existing process with the same name"
          verify_text:
            type: string
            description: "Optional app-specific text that must appear in the local/public HTTP response before a URL is reported"
          verify_wait_seconds:
            type: integer
            description: "Seconds to wait for verify_text checks"
          startup_wait_seconds:
            type: number
            description: "Seconds to wait after starting before checking whether the process exited"
          tunnel_protocol:
            type: string
            description: "cloudflared protocol; defaults to http2 for Docker/Linux networks where QUIC/UDP may be blocked"
          tunnel_attempts:
            type: integer
            description: "Transient public verification retry attempts; defaults to 3 and is capped at 5"
          tunnel_public_settle_seconds:
            type: number
            description: "Seconds to wait after cloudflared registration before public URL verification; defaults to 8 to avoid negative DNS caching"
          tail_chars:
            type: integer
            description: "Log tail size"
        required: [action]
---

# Service Expose

Use this skill when a user-created frontend, backend, API, WebSocket server, or preview app needs to keep running after the turn.

Default stance: when the user asks to create and start, deploy, preview, share, open, or make a service accessible, Cloudflare exposure is part of completion unless the user explicitly says local-only. A local URL, localhost port, PID, or background process alone is an intermediate state, not a final answer.

Prefer `service_expose` over one-off shell backgrounding for preview services because it persists service metadata, log paths, PIDs, local URLs, and tunnel URLs under `~/.spoon-bot/service-expose`.

## Actions

- `start`: run a service command in the background. Include `name`, `command`, and usually `cwd` plus `port` or `url`. For deploy/preview/share/access requests, pass `start_tunnel=true` in the same call unless the user explicitly asked for local-only.
- `tunnel` / `expose` / `start_tunnel`: start a Cloudflare Quick Tunnel for an existing service or an explicit local `url`.
- `status` / `list` / `inspect`: inspect running services and tunnel URLs.
- `logs`: return recent service or tunnel logs.
- `stop` / `stop_tunnel`: stop the background service or only its Cloudflare tunnel.

## Rules

Use `replace=true` only when the user wants to restart/replace the existing named service. Quick Tunnels use `cloudflared` from `CLOUDFLARED_PATH` or `PATH`; when it is missing on Linux/Windows, this skill can download the official latest Cloudflare binary into `~/.spoon-bot/service-expose/bin` unless `SPOON_BOT_AUTO_INSTALL_CLOUDFLARED=0`.

For generated services in sandbox/runtime environments, prefer one `service_expose` call with `action=start` and `start_tunnel=true`. Use `action=start` without a tunnel only for explicit local-only requests, internal preflight phases, or background helper services that the browser will not access. If a service was already started locally and the user still needs deployment/preview/share access, immediately call `service_expose` again with `action=tunnel` for that service before finalizing.

When creating a new preview service, pick a free port. Passing `port=0` asks this skill to choose an available port and inject it as `PORT` for the service command. Do not report a link until the local URL returns app-specific content. If a port is occupied, update the app command/config to use a free port and restart; do not treat an unrelated HTTP 200 on that port as success.

Before calling `service_expose` for generated code, complete the smallest local preflight that proves the service can launch: install declared runtime dependencies, run a syntax/build check for the entrypoint when the stack provides one, and fix failures before starting or exposing the service.

Cloudflare Quick Tunnels are for development previews. For production or stable hostnames, use a named Cloudflare Tunnel outside this quick-preview skill.

For WebSocket-only services, an HTTP `426 Upgrade Required` response or a successful TCP port probe is acceptable service reachability evidence when no `verify_text` was requested. Do not rewrite a working WebSocket server solely to satisfy an HTTP GET check unless the user also asked for a browser page.

The tunnel protocol defaults to `http2` because Docker and restricted Linux hosts often block or degrade QUIC/UDP. Override with `tunnel_protocol` or `SPOON_BOT_CLOUDFLARED_PROTOCOL` only when the environment requires another Cloudflare transport. If tunnel creation fails, inspect `service_expose` status/logs and retry through `service_expose`; do not bypass it with manual `cloudflared` shell commands.

Report a public URL only when the `service_expose` result has `success=true` and a non-null `public_url`, including nested tunnel results from `action=start` with `start_tunnel=true`. A trycloudflare URL that appears only in logs, `verification.url`, or a failed attempt is an unverified candidate and must not be presented as a usable link.

For browser apps, the public URL is not complete until every browser-required API, asset, and WebSocket endpoint is reachable from that public page. Prefer serving the frontend and API/WebSocket from one local service before exposing it; for example, serve `index.html` and upgrade WebSocket connections on the same port. If you use multiple local services, expose each browser-required service or rewrite/proxy the frontend so it does not point at localhost, loopback, or an unexposed local port. If a result contains `public_readiness.blocking=true`, resolve the listed missing public dependencies before finalizing.

If the tool reports Cloudflare Quick Tunnel rate limiting with `retry_after_seconds`, do not retry in the same turn. Wait for the cooldown or use a named/authenticated Cloudflare tunnel outside the quick-preview flow.
