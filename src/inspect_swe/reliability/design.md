# Reliability Design

This document explains the migrated reliability path: baseline runs, fault
runs, Inspect `.eval` logs, and the current `exec_command` observation fault.

## High-Level Flow

```mermaid
flowchart TD
    A[Inspect task] --> B[Reliability CLI]
    B --> C{Phase}
    C -->|baseline| D[Baseline runner]
    C -->|fault| E[Fault runner]
    D --> F[codex_cli default solver]
    E --> G[FaultEnvironment]
    G --> H[codex_cli with model filter]
    F --> I[Inspect eval]
    H --> I
    I --> J[.eval log]
```

## Exec Observation Fault Flow

```mermaid
sequenceDiagram
    participant M as Model
    participant C as Codex CLI
    participant T as exec_command
    participant F as FaultEnvironment

    M->>C: Request exec_command(cmd)
    C->>T: Run the requested command
    T-->>C: Real command observation
    C->>F: Next model call includes tool observation
    F->>F: Check latest ChatMessageTool(function="exec_command")
    F->>F: Sample deterministic fault decision
    alt fault applies
        F->>M: Generate with replaced bash-style error observation
    else no fault
        F->>M: Continue with original messages
    end
```

## Why Fault The Observation

The command itself is left unchanged. The fault is introduced after execution,
before the observation is sent back to the model. That gives a realistic failure
signal without breaking the tool protocol.

The injected observation currently looks like:

```text
Command: /bin/bash -lc <redacted>
Wall time: 0.0000 seconds
Process exited with code 254
Original token count: 8
Output:
bash: fork: Resource temporarily unavailable
```

In inspected GAIA traces, Codex can recover by issuing a later `exec_command`.

## Main Objects

- `ReliabilitySpec`: benchmark, agents, phases, seed, and concurrency.
- `BaselinePhaseConfig`: baseline run settings.
- `FaultPhaseConfig`: fault run settings and fault specs.
- `FaultSpec`: surface, mode, probability, severity, and optional target.
- `FaultEnvironment`: deterministic fault decisions and model/tool wrappers.

## Fault Rate

For `exec_observation_error`, probability is applied per new eligible
`exec_command` observation.

```mermaid
flowchart LR
    A[exec_command observation] --> B{Seen tool_call_id?}
    B -->|yes| C[Do not resample]
    B -->|no| D{Deterministic random < probability?}
    D -->|yes| E[Replace observation with bash error]
    D -->|no| F[Leave observation unchanged]
```

This means `p=0.5` does not guarantee exactly half of samples are faulted. It
means each eligible `exec_command` observation has an independent deterministic
chance to be faulted.

## Tested So Far

- Unit tests for deterministic decisions, exact `0.0` and `1.0` probability,
  exec-only scoping, and Codex default kwargs.
- GAIA Level 1 with `codex_cli` and GPT-5.4 at `p=0.8`.
- GAIA Level 1 with `codex_cli` and GPT-5.5 at `p=0.5`.

