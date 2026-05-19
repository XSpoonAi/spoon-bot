---
name: service_expose
description: Run user-created frontend or backend services in the background and optionally expose them through a Cloudflare Quick Tunnel.
version: 1.0.0
author: XSpoon Team
tags: [service, frontend, backend, cloudflare, tunnel, background]
triggers:
  - type: keyword
    keywords: [background service, run service, expose service, cloudflare tunnel, trycloudflare, frontend preview, backend preview]
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
            description: "start, tunnel, status, list, logs, stop, or stop_tunnel"
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
- `tunnel`: start a Cloudflare Quick Tunnel for an existing service or an explicit local `url`.
- `status` / `list`: inspect running services and tunnel URLs.
- `logs`: return recent service or tunnel logs.
- `stop` / `stop_tunnel`: stop the background service or only its Cloudflare tunnel.

## Rules

Use `replace=true` only when the user wants to restart/replace the existing named service. Quick Tunnels require `cloudflared` in `PATH` or `CLOUDFLARED_PATH`.

Cloudflare Quick Tunnels are for development previews. For production or stable hostnames, use a named Cloudflare Tunnel outside this quick-preview skill.
