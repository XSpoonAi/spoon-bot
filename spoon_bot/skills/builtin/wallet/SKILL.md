---
name: wallet
description: Use when working with spoon-bot's built-in EVM wallet, checking Neo X wallet readiness, or supporting skills that expect the legacy ~/.agent-wallet layout.
---

# Wallet

## Overview

`wallet` is a built-in spoon-bot skill. The wallet runtime is implemented in Python inside spoon-bot, so this skill is documentation and operating guidance only.

Do not install or bootstrap an external `openclaw/skills` wallet flow when this built-in skill is present.

## When to Use

Use this skill when:

- a task needs the agent wallet address, network, or compatibility paths
- a skill such as `joker-game-agent` expects the legacy `~/.agent-wallet` files
- the user asks whether a wallet already exists or was auto-created on startup
- the user wants to confirm which Neo X network is active

Do not use this skill to expose or print the raw private key unless the user explicitly asks for that secret.

## Private Key Safety (CRITICAL)

**NEVER** do any of the following:

- Read, cat, print, or echo the contents of `~/.agent-wallet/privatekey.tmp`
- Run `echo $PRIVATE_KEY`, `printenv PRIVATE_KEY`, or any command that would output the raw private key value
- Include raw private key hex values (0x + 64 hex chars) in your responses
- Pass the raw private key as a command-line argument

If a tool result or shell output contains what looks like a private key, **do not repeat it**. The value has been masked by the runtime. Continue operating normally — the wallet's signing capability is not affected by masking.

Skills that need the private key for signing transactions access it through the keystore file or the on-disk file directly. You do not need to read or relay the key yourself.

## Operating Rules

1. Treat the spoon-bot wallet runtime as the source of truth. Do not tell the user to run legacy shell onboarding scripts.
2. The canonical compatibility directory is `~/.agent-wallet`. spoon-bot keeps this layout so existing skills can work without modification.
3. On startup, spoon-bot auto-creates a wallet when one does not exist, unless `SPOON_BOT_WALLET_AUTO_CREATE=false`.
4. The default network is `neox` (Neo X Mainnet). `neox_testnet` is also supported.
5. If the operator already provided `WALLET_ADDRESS`, `RPC_URL`, or `ETH_RPC_URL`, treat those values as higher priority than the auto-created defaults.
6. Do not delete, rotate, or replace wallet files unless the user explicitly requests that action.

## What To Check

For wallet readiness, prefer these checks:

- `~/.agent-wallet/state.env` for address, chain id, RPC URL, and compatibility paths
- `~/.agent-wallet/keystore.json` and `~/.agent-wallet/pw.txt` for local wallet presence
- a consuming CLI such as `node skills/joker-game-agent/cli/index.js wallet` when verifying downstream compatibility

Expected default files:

- `~/.agent-wallet/keystore.json`
- `~/.agent-wallet/pw.txt`
- `~/.agent-wallet/state.env`

## User-Facing Behavior

When the wallet was auto-created during startup, tell the user plainly that spoon-bot created a local wallet automatically and keep the summary short:

- wallet address
- active network
- wallet directory

If the user asks for wallet details, prefer sharing the address and network first. Only reveal secrets such as the private key when the user explicitly asks for them.

## Common Mistakes

- Reading or printing `privatekey.tmp` or `$PRIVATE_KEY` — this exposes the key into the conversation and may cause the LLM to refuse further wallet operations
- Treating this built-in skill as an external installable wallet package
- Reintroducing shell-script onboarding after the Python wallet runtime already exists
- Overwriting operator-supplied wallet environment variables during bootstrap
- Moving the wallet away from `~/.agent-wallet` and breaking compatibility with existing skills
