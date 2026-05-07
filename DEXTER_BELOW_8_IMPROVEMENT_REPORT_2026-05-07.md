# DEXTER Below-8 Improvement Report (Detailed)

Date: 2026-05-07
Purpose: Deep corrective report for all areas scoring below 8/10.
Method: Truth-first diagnosis -> root causes -> exact fixes -> measurable targets.

## Scope
This report covers these categories only:
- Reliability (runtime): 6.8/10
- Safety/security controls: 6.5/10
- Testing maturity: 6.9/10
- Observability/diagnostics: 6.7/10
- Product UX readiness: 3.8/10
- Deployment/release readiness: 3.2/10
- Commercial/legal readiness: 3.5/10
- Productization score: 4.0/10
- Sell-now readiness score: 3.6/10

----------------------------------------

## 1) Reliability (Runtime) - Current 6.8/10

### What you have now
- Async orchestration exists and is reasonably clean.
- State transitions and watchdog behavior exist.
- Multi-LLM fallback exists (Gemini -> Groq -> Ollama).
- Tool execution timeout exists.
- CPU fallback exists for STT.

### What is wrong
1. Single-process fragility.
- One bad dependency state or unrecoverable exception can degrade entire assistant quality.

2. No production supervisor.
- There is no process manager/service watchdog outside app runtime.

3. Hardware variability risk.
- CUDA assumptions are handled partially, but no explicit compatibility profile strategy for customer devices.

4. External dependency brittleness.
- Browser, audio stack, network, and model providers are variable, but failure recovery strategy is still basic.

### How to fix (detailed)
1. Add process-level supervision.
- Build a launcher/supervisor process that restarts Dexter on non-zero exit and tracks restart frequency.
- Add exponential restart backoff and crash-loop protection.

2. Introduce capability profiles.
- Profile A: CPU-safe baseline.
- Profile B: CUDA-accelerated.
- Auto-detect at startup and persist selected profile in runtime diagnostics.

3. Improve fault isolation.
- Wrap high-risk integrations (audio input, STT inference, browser operations) in isolated worker boundaries.
- Use strict timeout + fallback per boundary.

4. Add resilience policies.
- Define provider circuit-breakers with explicit open/half-open/closed semantics.
- Add retry policies with jitter and max retry budgets.

### Success criteria
- Crash-free sessions over 8-hour soak test >= 99%.
- Recovery from provider outage < 5 seconds to next available provider.
- Startup success rate across mixed hardware matrix >= 95%.

### Target score after fixes
- 8.0 to 8.4/10

----------------------------------------

## 2) Safety/Security Controls - Current 6.5/10

### What you have now
- Tool schema validation exists.
- Tool timeout exists.
- Path restrictions exist.
- Basic unsafe argument pattern filtering exists.
- Power actions have confirmation gate.

### What is wrong
1. No formal threat model.
- Security controls exist, but no written model of threats and controls.

2. Prompt-injection defense is incomplete.
- Current controls are mostly execution-side, not full adversarial instruction handling.

3. Sensitive action policy is uneven.
- Confirmations exist for some actions, but not full risk-tier governance.

4. Secrets and data governance are not product-grade.
- No formal policy package for mic/screen/clipboard data and retention behavior.

### How to fix (detailed)
1. Create security policy tiers.
- Tier 0: read-only low risk.
- Tier 1: local state changes.
- Tier 2: destructive/system actions.
- Force explicit user confirmation for Tier 2 and for selected Tier 1.

2. Add intent firewall.
- Build a pre-execution policy engine that evaluates tool call context + source + risk score.
- Reject/hold calls that violate policy constraints.

3. Prompt-injection test suite.
- Build adversarial prompts for web content, clipboard content, and transcribed speech.
- Verify no unauthorized tool execution occurs.

4. Secrets/data governance.
- Define secret handling lifecycle.
- Define data retention policy for transcripts/screenshots/logs/memory vectors.
- Add user-facing privacy controls and deletion paths.

### Success criteria
- 0 critical unauthorized executions in adversarial suite.
- 100% high-risk actions gated by explicit confirmation.
- Published and versioned security policy + privacy controls.

### Target score after fixes
- 7.8 to 8.3/10

----------------------------------------

## 3) Testing Maturity - Current 6.9/10

### What you have now
- Tests exist and pass under tests folder.
- Streaming behavior and tool routing are tested.
- Preflight audit script exists.

### What is wrong
1. Test discovery hygiene issue.
- Root pytest run breaks due to text artifact collection.

2. Missing reliability test classes.
- No soak/stress/fault-injection and noisy audio regression suites.

3. Missing end-to-end acceptance tests.
- Real user journeys are not sufficiently automated.

4. Missing quality gates in CI.
- No automated branch protection pipeline.

### How to fix (detailed)
1. Fix test discovery immediately.
- Add pytest configuration with explicit testpaths and patterns.

2. Expand test pyramid.
- Unit: parser/router/policy boundaries.
- Integration: tool executor + provider fallback.
- E2E: speech in -> action -> response out user stories.
- Non-functional: load, soak, crash recovery, noise robustness.

3. Add CI gating.
- Mandatory lint, type-check, unit, integration, and selected E2E smoke.

4. Add release validation suite.
- Minimum release criteria: pass matrix (CPU/CUDA, with/without provider keys).

### Success criteria
- Root pytest pass is stable.
- >= 80% critical-path coverage.
- Soak test pass over 8 hours.
- Release gate pass required before tagging.

### Target score after fixes
- 8.0 to 8.5/10

----------------------------------------

## 4) Observability/Diagnostics - Current 6.7/10

### What you have now
- Correlation-ID logging exists.
- Rotating logs exist.
- Metrics collector exists with provider/latency snapshots.

### What is wrong
1. No centralized diagnostics dashboard.
- Troubleshooting still requires log-file/manual analysis.

2. No alerting/SLO discipline.
- Latencies and failures are collected but not operated with formal thresholds.

3. No customer diagnostics bundle.
- Supportability is weak for non-technical users.

### How to fix (detailed)
1. Define SLOs and error budgets.
- Example SLOs: time-to-first-response, command success rate, crash rate, provider failover time.

2. Add health endpoint + diagnostics pack.
- Collect config profile, provider status, recent errors, latency histograms, audio device state.
- One-click export for support.

3. Add event-level lifecycle traces.
- Standard lifecycle event IDs per utterance for full traceability.

4. Build operator panel (even minimal local web panel first).
- Real-time status for pipeline state, providers, tools, and failure causes.

### Success criteria
- Mean time to diagnose top 10 failures reduced by >= 50%.
- SLO reporting available per release.
- Support bundle generation < 30 seconds.

### Target score after fixes
- 8.0 to 8.3/10

----------------------------------------

## 5) Product UX Readiness - Current 3.8/10

### What you have now
- Functional assistant core with voice + automation.
- No end-user app UX shell.

### What is wrong
1. No GUI/tray UX.
- No polished user-facing app for mainstream users.

2. No onboarding and discoverability.
- No guided setup for mic, wake behavior, providers, permissions.

3. No user trust UI.
- No clear visibility of what Dexter is doing, when, and why.

4. No safe controls for non-technical users.
- Missing easy controls: mute, pause listening, permission review, memory clear.

### How to fix (detailed)
1. Build desktop shell (Phase 1).
- System tray app, compact status panel, settings panel, command history.
- Show state transitions (listening/processing/speaking/error).

2. Build onboarding flow.
- Device check (mic/speaker), key validation, capability profile detection (CPU/CUDA), permission preferences.

3. Add user trust controls.
- Tool call preview for risky actions.
- Privacy panel (logs, screenshots, memory retention controls).

4. Add UX polish.
- Fast feedback chimes, clear failure messages, retry suggestions, fallback transparency.

### Success criteria
- New user setup completion > 85% without developer help.
- First-task success in < 5 minutes from install.
- User-reported trust/comprehension score >= 8/10.

### Target score after fixes
- 7.2 to 8.0/10

----------------------------------------

## 6) Deployment/Release Readiness - Current 3.2/10

### What you have now
- Code can run in local dev environment.
- No formal release machinery.

### What is wrong
1. No installer.
- No MSIX/Inno/NSIS/WiX packaging path.

2. No signed releases.
- No code signing plan; trust and Windows friction issues.

3. No update channel.
- No versioned update mechanism.

4. No CI/CD.
- No automated validation and release pipeline.

### How to fix (detailed)
1. Add build system.
- Reproducible build scripts and pinned dependency strategy.

2. Add installer path.
- Build Windows installer with prerequisite checks and rollback-safe install/uninstall.

3. Add signing and update strategy.
- Code signing for binaries.
- Auto-update channel with staged rollout.

4. Add CI/CD release flow.
- Build/test/package/sign/publish pipeline with release notes generation.

### Success criteria
- One-command reproducible release build.
- Signed installer artifacts produced per release.
- In-place upgrade success rate >= 98%.

### Target score after fixes
- 7.5 to 8.2/10

----------------------------------------

## 7) Commercial/Legal Readiness - Current 3.5/10

### What you have now
- Clear technical intent to commercialize.
- No complete compliance packaging yet.

### What is wrong
1. No formal legal pack.
- Missing EULA, privacy policy, data-processing disclosure, support terms.

2. License-risk visibility incomplete.
- Needs full dependency and model-provider terms mapping.

3. No commercial operations plan.
- Pricing, support SLA tiers, and incident handling are undefined.

### How to fix (detailed)
1. Build legal baseline package.
- EULA, privacy policy, terms, data retention/deletion commitments.

2. Build SBOM + license compliance review.
- License inventory for all dependencies and transitive dependencies.
- Confirm allowed commercial redistribution and usage terms.

3. Provider contract strategy.
- Define clear SKU behavior by provider availability and pricing risk.
- Add quota/degradation policy per plan.

4. Build support operations model.
- Incident flow, support tooling, escalation tiers, diagnostics intake.

### Success criteria
- Compliance checklist fully green before first paid release.
- Legal docs published and versioned.
- Support SLA and incident runbook operational.

### Target score after fixes
- 7.0 to 8.0/10

----------------------------------------

## 8) Productization (Current 4.0/10) - What exactly is missing

### Current reality
The engineering core is strong, but product envelope is weak.

### Missing layers
- UX layer: app shell + onboarding + trust controls
- Ops layer: release, update, diagnostics, support
- Governance layer: legal, privacy, licensing, policy

### Corrective strategy
- Build these 3 layers in parallel tracks, not sequentially.
- Assign owners and weekly milestones per layer.

### Target score after fixes
- 7.5 to 8.2/10

----------------------------------------

## 9) Sell-now Readiness (Current 3.6/10) - Brutal Truth

### Why this is low
- You have an engine, not a polished product.
- You can demo strongly, but mainstream paid deployment will currently generate high support burden and trust friction.

### Minimum bar to move above 7/10
1. Stable signed installer + updater
2. End-user GUI and onboarding
3. Security policy and privacy controls
4. Legal documentation and license confidence
5. CI/CD quality gates + release criteria
6. Support diagnostics pipeline

### Target score after fixes
- 7.0 to 7.8/10

----------------------------------------

## 10) Detailed "What We Are Doing Wrong" Summary

1. Over-investing in core features before shipping envelope.
- Core is stronger than deployment/product infrastructure.

2. Under-investing in release engineering.
- Without build/sign/update/rollback discipline, commercialization fails even if AI works.

3. Under-investing in trust UX.
- Users need visibility and control for mic/screen/automation behavior.

4. Under-investing in formal risk/governance artifacts.
- Security and legal controls must be explicit and auditable before sales.

5. Not running a true production matrix.
- Real customer hardware/network/provider permutations are not yet covered.

----------------------------------------

## 11) 90-Day Corrective Program (Detailed)

### Days 0-30: Foundation Hardening
- Fix test discovery and establish CI quality gates.
- Add process supervisor and crash diagnostics bundle.
- Define security tiers and apply policy checks.
- Deliver release checklist v1.

Expected score movement:
- Reliability: +0.6
- Testing: +0.8
- Observability: +0.7

### Days 31-60: Product Envelope
- Ship tray + settings + onboarding UX.
- Add privacy controls and user-visible action logs.
- Add installer prototype and signed internal builds.

Expected score movement:
- UX readiness: +2.5 to +3.2
- Deployment readiness: +2.0 to +2.8

### Days 61-90: Commercial Readiness
- Finalize legal pack and SBOM/license review.
- Introduce updater and release channels.
- Run pilot cohort with support runbook.

Expected score movement:
- Commercial/legal: +2.5 to +3.5
- Sell-now readiness: +2.5 to +3.2

----------------------------------------

## 12) Final Improvement Projection
If the above program is executed with discipline:
- Reliability: 6.8 -> 8.1
- Security: 6.5 -> 8.0
- Testing: 6.9 -> 8.2
- Observability: 6.7 -> 8.1
- UX readiness: 3.8 -> 7.4
- Deployment readiness: 3.2 -> 7.8
- Commercial/legal: 3.5 -> 7.3
- Productization: 4.0 -> 7.8
- Sell-now readiness: 3.6 -> 7.2

This is realistic, but only with strong execution discipline and dedicated productization effort.
