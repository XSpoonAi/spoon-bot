# spoon-bot Agent Instructions

## No Case-Specific Routing

- Do not add prompt-specific routes, regex classifiers, hidden forced prompts, or one-off workflow branches just to make a replay, demo, or single user prompt pass.
- Preserve the user's original prompt. Do not rewrite it into a different task, inject a case-specific plan, or steer it through a hardcoded skill/game/service path.
- When a prompt fails, fix the reusable layer underneath: skill metadata, dynamic tool activation, tool contracts, structured request facts, environment handling, provider errors, or loop/tool orchestration.
- Regression tests should assert generic behavior and contracts, not exact prompt strings or named demo repositories unless the feature itself owns that fixture.
- Temporary diagnostic logic must stay out of committed production paths. If a narrow probe is needed for debugging, remove it before commit.
