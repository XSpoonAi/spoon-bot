---
name: service_expose
description: Run user-created frontend or backend services in the background and optionally expose them through a Cloudflare Quick Tunnel. Use for preview links, localhost URLs, public/trycloudflare links, WebSocket apps, and service URL verification.
version: 1.0.0
author: XSpoon Team
tags: [service, frontend, backend, cloudflare, tunnel, background]
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
          tail_chars:
            type: integer
            description: "Log tail size"
        required: [action]
---

# Service Expose

Use this skill when a user-created frontend or backend needs to keep running after the turn and be reachable through Cloudflare.

Prefer `service_expose` over one-off shell backgrounding for preview services because it persists service metadata, log paths, PIDs, local URLs, and tunnel URLs under `~/.spoon-bot/service-expose`.

## Actions

- `start`: run a service command in the background. Include `name`, `command`, and usually `cwd` plus `port` or `url`.
- `tunnel` / `expose` / `start_tunnel`: start a Cloudflare Quick Tunnel for an existing service or an explicit local `url`.
- `status` / `list` / `inspect`: inspect running services and tunnel URLs.
- `logs`: return recent service or tunnel logs.
- `stop` / `stop_tunnel`: stop the background service or only its Cloudflare tunnel.

## Rules

Use `replace=true` only when the user wants to restart/replace the existing named service. Quick Tunnels use `cloudflared` from `CLOUDFLARED_PATH` or `PATH`; when it is missing on Linux/Windows, this skill can download the official latest Cloudflare binary into `~/.spoon-bot/service-expose/bin` unless `SPOON_BOT_AUTO_INSTALL_CLOUDFLARED=0`.

When creating a new preview service, pick a free port. Passing `port=0` asks this skill to choose an available port and inject it as `PORT` for the service command. Do not report a link until the local URL returns app-specific content. If a port is occupied, update the app command/config to use a free port and restart; do not treat an unrelated HTTP 200 on that port as success.

Cloudflare Quick Tunnels are for development previews. For production or stable hostnames, use a named Cloudflare Tunnel outside this quick-preview skill.
