# Spoon-Bot 多会话支持改造方案

更新日期：2026-05-07

## 当前进度

- 已完成阶段 1：新增 `SessionRuntimeRegistry` / `SessionRuntime`，并在 gateway 启动、关闭路径中接入 registry。
- 已完成阶段 2：HTTP chat、SSE streaming、WebSocket chat、`session.switch`、`session.list` 已切换到 session runtime 主路径，并在事件 payload 中补充 `session_key`。
- 已完成阶段 3：Channels 与 Cron 已按 `session_key` 获取独立 runtime，并移除这些主路径对 `primary-agent` 全局 runner lock 的依赖。
- 已完成阶段 4 主体：registry 支持 idle runtime 回收、active runtime 上限与 LRU 淘汰；REST `POST /v1/sessions/{session_key}/close` 与 WebSocket `session.close` 可关闭内存 runtime 并保留持久化历史。
- 已补测试：`tests/test_session_runtime_registry.py` 覆盖 runtime 复用、并发创建去重、关闭时保存会话、idle 回收、LRU 淘汰；`tests/test_multi_session_runtime_integration.py` 覆盖 REST 不同会话并发，以及同一 WebSocket 连接内不同 `session_key` 的流式并发。
- 已完成阶段 5 观测主体：REST / WebSocket `agent.status` 暴露 `runtime_metrics`，包含 active / running / idle、创建数、关闭数、idle 回收数、LRU 淘汰数与配置值。
- 已完成 API 文档细化：README 与 `docs/api.md` 已记录多会话 runtime 语义、WebSocket 并发流式行为、事件字段与部署验证注意事项。
- 后续可继续加强：真实模型/真实工具下的长流式压力测试，以及跨进程部署场景的观测面板。

## 1. 背景与目标

当前 spoon-bot 已经具备按 `session_key` 持久化历史的能力：

- `spoon_bot/session/manager.py` 提供 `SessionManager` 与内存缓存。
- `spoon_bot/session/store.py` 提供 file / sqlite / postgres 后端。
- REST 与 WebSocket 已暴露 `session.list`、`session.switch`、`session.clear`、`session.export`、`session.import`、`session.search` 等入口。

但当前 gateway 运行态仍然以单个全局 `AgentLoop` 为核心。HTTP、WebSocket、渠道、cron 在处理请求时会临时切换 `agent._session` 与 `agent.session_key`，同时使用 `primary-agent` 全局 runner lock 串行化执行。这种方式可以避免数据竞争，但不是真正的多会话运行时：

- 不同会话无法并发执行。
- 同一个全局 agent 的可变运行态会在不同 session 之间来回切换。
- WebSocket 的当前会话只保存在连接对象上，缺少统一的 session runtime registry。
- 未来要支持多用户、多 tab、多渠道、cron 并发时，隔离边界不够清晰。

本次改造目标：

1. 让每个逻辑 `session_key` 拥有独立的 agent runtime 状态。
2. 共享配置、工具定义、技能市场、持久化 store 等只读或可安全共享资源。
3. 同一 session 内串行执行，不同 session 之间允许并发执行。
4. 保持现有 REST / WebSocket API 兼容，前端可继续使用 `session_key`。
5. 为后续横向扩展、session idle 回收、会话级状态查询留下接口。

## 2. 设计原则

- **Session 是运行时隔离边界**：`session_key` 不只是历史文件名，也是 runtime owner key。
- **持久化历史是权威数据源**：runtime 可被回收，恢复时从 `SessionManager` 注入历史。
- **锁跟着可变对象走**：同一 session 的 runner 需要锁；不同 session 的 runner 不共享全局执行锁。
- **连接不等于会话**：一条 WebSocket 可以切换 session；session runtime 生命周期独立于连接。
- **兼容优先**：`default` session 继续存在；老客户端不传 `session_key` 时行为不变。

## 3. 目标架构

新增一个会话运行时管理层，建议命名为 `SessionRuntimeRegistry`。

```text
Gateway app
  |-- SessionRuntimeRegistry
  |   |-- shared config snapshot
  |   |-- shared SessionManager
  |   |-- session_key -> SessionRuntime
  |   `-- idle cleanup task
  |
  |-- ConnectionManager
  |   `-- connection_id -> current session_key
  |
  `-- REST / WS / Channels / Cron
      `-- resolve session_key -> acquire SessionRuntime -> run

SessionRuntime
  |-- session_key
  |-- AgentLoop instance
  |-- runner_lock
  |-- status: idle / running / cancelling / closed
  |-- last_used_at
  `-- active_task metadata
```

关键变化：

- gateway 启动时不再只创建一个全局 `AgentLoop` 给所有请求直接使用。
- gateway 启动时创建 `SessionRuntimeRegistry`，内部保留一个默认 session runtime，或按需 lazy 创建。
- REST / WS / Channels / Cron 统一通过 registry 获取对应 session 的 runtime。
- 每个 runtime 内部的 `AgentLoop.session_key` 固定为该 runtime 的 `session_key`，请求期间不再反复切换全局 agent。

## 4. 主要改动点

### 4.1 新增 SessionRuntimeRegistry

建议新增文件：

- `spoon_bot/runtime/session_registry.py`

核心接口：

```python
class SessionRuntimeRegistry:
    async def get_or_create(self, session_key: str) -> SessionRuntime: ...
    async def get(self, session_key: str) -> SessionRuntime | None: ...
    async def list(self) -> list[SessionRuntimeInfo]: ...
    async def close(self, session_key: str) -> bool: ...
    async def close_all(self) -> None: ...
```

`SessionRuntime` 建议包含：

```python
@dataclass
class SessionRuntime:
    session_key: str
    agent: AgentLoop
    lock: asyncio.Lock
    created_at: datetime
    last_used_at: datetime
    active_task_id: str | None = None
```

创建 runtime 时复用 gateway 启动时解析好的 agent config，并注入同一个 `SessionManager`：

- `session_manager` 共享，保证所有 session 使用同一 store。
- 每个 `AgentLoop` 使用自己的 `_agent`、memory、tool execution context、context builder。
- 工具注册表可先按 AgentLoop 当前初始化方式各自创建，后续再优化共享只读 tool catalog。

### 4.2 改造 gateway app 全局状态

当前：

- `get_agent()` 返回单个全局 `AgentLoop`。
- `get_agent_execution_lock()` 返回 `primary-agent` 全局锁。

建议：

- 新增 `set_session_runtime_registry()` / `get_session_runtime_registry()`。
- `get_agent()` 保留为兼容入口，返回 default runtime 的 agent，逐步减少新代码使用。
- `get_agent_execution_lock()` 标记为兼容路径，新请求处理改用 `runtime.lock`。

涉及文件：

- `spoon_bot/gateway/app.py`
- `spoon_bot/gateway/server.py`

### 4.3 改造 HTTP Agent API

当前 `spoon_bot/gateway/api/v1/agent.py` 中会：

1. 解析 `session_key`
2. 获取 session lock
3. 获取全局 agent lock
4. `_switch_session(agent, session_key)`
5. 调用 `agent.process()` / `agent.stream()`

改造后：

1. 解析 `session_key`
2. `runtime = await registry.get_or_create(session_key)`
3. `async with runtime.lock`
4. 调用 `runtime.agent.process(..., session_key=session_key)` 或直接调用固定 session agent

注意点：

- 去掉请求级 `_switch_session()` 对全局 agent 的依赖。
- SSE streaming 生命周期内持有该 session runtime lock。
- 同一 session 的并发请求仍串行；不同 session 可以并发。

### 4.4 改造 WebSocket Handler

当前 `spoon_bot/gateway/websocket/handler.py` 中：

- `Connection.session_key` 保存当前 session。
- `session.switch` 只改连接对象上的 `session_key`。
- `agent.chat` 会临时切换全局 agent session。
- 当前 handler 只能跟踪本连接上的 `_current_task`。

改造后：

- `session.switch`：
  - 校验 session_key。
  - 通过 registry lazy 创建目标 runtime。
  - 更新连接 `session_key`。
  - 返回 runtime status 与 session metadata。
- `agent.chat` / `chat.send`：
  - 从 params 或连接当前值解析 `session_key`。
  - 获取对应 runtime。
  - 在 `runtime.lock` 内执行。
  - emitted event 中统一带上 `session_key`，方便前端多 tab 过滤。
- `agent.cancel`：
  - 优先取消当前连接发起的任务。
  - 后续可扩展为按 `session_key` 取消 session runtime 的 active task。

### 4.5 改造 Channels 与 Cron

当前渠道和 cron 已经有独立的 session_key 生成逻辑，但仍通过同一个 agent runner 执行。

涉及文件：

- `spoon_bot/channels/manager.py`
- `spoon_bot/cron/executor.py`

改造方式：

- 渠道消息根据 message.session_key 获取对应 runtime。
- cron job 根据 job.session_key 获取对应 runtime。
- `get_runner_lock("primary-agent")` 改为使用 `runtime.lock`。
- 对于需要清理历史的 job，继续调用共享 `SessionManager`，但只操作目标 session。

## 5. Session 生命周期

建议状态：

```text
created -> idle -> running -> idle
                 -> cancelling -> idle
idle -> closing -> closed
```

生命周期策略：

- 默认 session 在 gateway 启动时可预热，也可 lazy 创建。
- 非 default session 第一次访问时 lazy 创建。
- runtime idle 超过配置时间后关闭内存态，但保留持久化历史。
- 关闭 runtime 前确保：
  - 没有 active task，或先取消并等待。
  - 当前 `_session` 已保存。
  - 释放 HTTP client / tool process / terminal 等 session 级资源。

建议配置：

```text
SPOON_BOT_SESSION_RUNTIME_IDLE_SECONDS=1800
SPOON_BOT_SESSION_RUNTIME_MAX_ACTIVE=64
SPOON_BOT_SESSION_RUNTIME_PREWARM_DEFAULT=true
```

## 6. 兼容性与迁移

保持兼容：

- REST `POST /v1/agent/chat` 的 `session_key` 字段不变。
- WebSocket `session.switch` / `agent.chat` 参数不变。
- `GET /v1/sessions` 仍从 `SessionManager` 列持久化 session。
- `default` session 行为不变。
- file / sqlite / postgres store 不需要变更 schema。

需要补充：

- `agent.status` 增加 `active_sessions` 与当前 session runtime 状态。
- `session.list` 可额外返回 `runtime_status`、`active_task_id`、`last_used_at`。
- 事件 payload 增加 `session_key`，老客户端忽略即可。

## 7. 风险点

1. **资源占用上升**：每个 session 一个 AgentLoop，会带来更多 tool registry、memory、MCP client 等对象。需要 idle 回收和最大 active 数限制。
2. **工具副作用冲突**：不同 session 可能同时操作同一个 workspace 文件。现有 `WorkspaceFSService` 有路径锁，但 shell/tool 层仍要评估是否需要 workspace 级写锁。
3. **共享服务线程安全**：SessionManager 已有 RLock，但部分 tool、skill、memory、MCP 对象如果共享，需确认并发安全。第一阶段建议 runtime 内独立创建可变对象。
4. **取消语义**：WebSocket 当前取消偏连接级；多 session 后应明确连接级取消与 session 级取消的边界。
5. **子代理与历史搜索**：subagent manager 当前共享 SessionManager，改造后要确认 default session resolver 指向当前 runtime，而不是全局 agent。

## 8. 实施步骤

### 阶段 1：引入 registry，但保持串行兼容

- 新增 `SessionRuntimeRegistry` 与 `SessionRuntime`。
- gateway 启动时创建 registry。
- default runtime 包装现有 AgentLoop。
- HTTP / WS 先通过 registry 获取 runtime，但仍保留全局锁兜底。
- 补单测：同 session 返回同一个 runtime，不同 session 返回不同 runtime。

### 阶段 2：请求路径切换到会话级 runtime

- HTTP chat / SSE 改用目标 runtime。
- WebSocket chat 改用目标 runtime。
- 移除请求期间 `_switch_session()` 的主路径依赖。
- 事件里补 `session_key`。
- 补并发测试：两个 session 同时执行不会互相切换 `_session`。

### 阶段 3：Channels / Cron 接入 registry

- 渠道消息按 session_key 获取 runtime。
- cron job 按 session_key 获取 runtime。
- 移除 `primary-agent` 锁在这些路径上的使用。
- 补测试：渠道 A/B 不同 session 的历史分别落入不同 session。

### 阶段 4：生命周期治理

- 增加 idle cleanup task。
- 增加 active runtime 上限与 LRU 淘汰。
- 增加 `session.close` 或 REST 管理接口，可选。
- 补测试：idle runtime 被关闭后再次访问可从历史恢复。

### 阶段 5：文档与观测

- 更新 `docs/api.md` 与 WebSocket 文档。
- 增加 runtime metrics：
  - active session count
  - per-session running / idle
  - evicted runtime count
  - session creation latency
- 日志统一带 `session_key`、`runtime_id`、`task_id`。

## 9. 测试计划

单元测试：

- `SessionRuntimeRegistry.get_or_create()` 并发同 key 只创建一个 runtime。
- 不同 session 的 AgentLoop `_session` 对象互不相同。
- runtime close 会保存当前 session。
- idle cleanup 只回收空闲 runtime，不打断 active task。

HTTP 测试：

- 两个不同 `session_key` 连续调用 chat，历史分别持久化。
- 两个不同 `session_key` 并发调用，互不污染。
- SSE streaming 的 done metadata 带正确 `session_key`。

WebSocket 测试：

- `session.switch` 后 `agent.chat` 写入新 session。
- 同一连接切回旧 session 可恢复旧历史。
- 两条连接分别使用不同 session，同时 streaming 不串线。
- `agent.cancel` 只影响当前连接或指定 session 的 active task。

渠道 / Cron 测试：

- Telegram / Discord / Feishu 生成的 session_key 对应独立 runtime。
- cron isolated/current/session 三种 target mode 行为保持兼容。

回归测试：

- `tests/test_session_persistence.py`
- `tests/test_websocket_runtime.py`
- `tests/test_gateway_ws_bugs.py`
- `tests/test_cron_executor.py`
- `tests/test_discord_channel.py`
- `tests/test_feishu_channel.py`

## 10. 验收标准

- 老客户端不传 `session_key` 时仍走 `default` session。
- 任意两个不同 `session_key` 的请求不会修改彼此的 `agent._session`。
- 同一 `session_key` 的并发请求仍串行执行，历史顺序稳定。
- 不同 `session_key` 的请求可以并发执行。
- WebSocket 与 REST 对同一 session 读写同一份持久化历史。
- runtime 被回收后再次访问，可以从持久化历史恢复上下文。

## 11. 当前实现说明

当前 gateway 使用 `SessionRuntimeRegistry` 管理内存运行态。每个 live `session_key` 对应一个 `SessionRuntime`，其中包含固定绑定到该 session 的 `AgentLoop`、会话级执行锁、`active_task_id`、创建时间和最近使用时间。

运行时隔离规则：

- 同一 `session_key` 内部通过 `runtime.lock` 串行执行。
- 不同 `session_key` 使用不同 runtime，可并发执行 HTTP、SSE、WebSocket、channel 和 cron 请求。
- `default` runtime 作为兼容入口保留；未传 `session_key` 的旧客户端仍使用 `default`。
- 非 default runtime 必须能从默认 agent 派生创建参数并 clone 新 agent；如果无法 clone，registry 会拒绝复用 default agent，以避免不同会话共享可变运行态。

WebSocket 行为：

- 一条 `/v1/ws` 连接可以连续发送多个 `chat.send` / `agent.chat` 请求。
- 请求参数里的 `session_key` 优先于连接当前 session；如果未传，则使用连接当前 session。
- 同一连接内，不同 `session_key` 的流式请求会作为独立后台任务运行，可以交错发送 `agent.stream.chunk`。
- `agent.thinking`、`agent.stream.chunk`、`agent.stream.done`、`agent.complete`、`agent.error`、`agent.cancelled` 都应携带 `task_id`、`request_id`、`session_key` 和 `trace_id`，方便上游按会话和请求归属路由。
- `chat.cancel` 可通过 `session_key` 取消指定会话的当前任务；不传时取消当前连接/session 的当前任务。

观测与验证：

- `GET /v1/agent/status` 与 WebSocket `agent.status` 返回 `runtime_metrics`，用于确认 active/running/idle runtime 数量。
- 回归测试 `tests/test_multi_session_runtime_integration.py::test_websocket_different_sessions_stream_on_same_connection_concurrently` 验证同一 WebSocket 连接内两个不同 session 能同时进入 stream，并都能收到 chunk。
- 部署到镜像环境时，确认运行中的 sandbox/container 确实使用包含本改造的镜像。若日志中的 `agent.thinking` 缺少 `request_id` 或 `session_key`，通常说明运行的是旧镜像或旧进程，而不是日志截断。
