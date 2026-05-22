# DEXTER Improvement Plan

**Date:** 2026-05-22
**Purpose:** Concrete engineering roadmap to move DEXTER from a strong personal assistant into a stable, distributable product.

## 1. Goals

DEXTER already has a credible runtime, but the next stage is about productizing the architecture rather than adding more surface area.

Primary goals:
- make the assistant more reliable under real-world desktop conditions
- harden the RAG and document-resolution experience
- reduce control-flow complexity in the turn pipeline
- improve human-like continuity across sessions
- prepare for GUI and MCP consumers without weakening safety
- make GPU/CUDA readiness visible and diagnosable
- keep automation safe, predictable, and recoverable

## 2. Priority Order

### Phase 1: Stabilize the runtime core
- Refactor the pipeline in `core/pipeline.py` into smaller turn-stage handlers.
- Separate listening, activation, retrieval, response generation, and speech playback.
- Replace the single global turn timeout with stage-specific time budgets.
- Add more explicit failure reporting around turn cancellation and fallback routing.
- Keep the current behavior, but make the control flow easier to test and reason about.

### Phase 2: Finish the document and RAG split
- Keep direct file reads separate from project-name resolution.
- Treat project lookup as a first-class workflow with its own confidence rules.
- Make project context persistent instead of in-memory only.
- Move document and RAG thresholds into config.
- Add recovery behavior for ambiguous or wrong-file selections.
- Add tests for project-name collisions and low-confidence lookups.

### Phase 3: Add durable memory and behavior continuity
- Persist the current project slot across restarts.
- Add preference memory for verbosity, tone, and correction history.
- Feed explicit user corrections back into future retrieval and phrasing.
- Add a clearer distinction between quick answers and deep explanations.
- Make the assistant feel continuous across sessions rather than stateless.

### Phase 4: Harden automation and desktop control
- Improve `pyautogui` focus handling with configurable wait times.
- Verify foreground state before and after typing or shortcut execution.
- Add clearer health and failure reporting for automation availability.
- Treat GUI automation as a recoverable subsystem, not a silent best-effort layer.
- Add a small automation audit tool for local reliability checks.

### Phase 5: Prepare GUI and MCP surfaces
- Use the event bus as the main bridge for UI and observer clients.
- Expose runtime state, health, and current project context through stable read-only surfaces.
- Add recent event history or replay support.
- Make MCP and native tool execution share the same policy layer.
- Build the GUI only after the state and event contract is stable.

### Phase 6: Improve GPU and hardware diagnostics
- Add boot-time GPU diagnostics for CUDA availability and memory pressure.
- Make embedding and transcription batch sizes react to hardware capability.
- Add a stronger CPU-only fallback mode.
- Report low-power mode choices clearly so users know what changed.
- Monitor memory and GPU behavior in long-running sessions.

## 3. Immediate Engineering Work

The next 1 to 2 weeks should focus on the highest-return work:
- refactor `core/pipeline.py` to reduce turn-loop complexity
- persist session/project context
- make RAG and document confidence thresholds configurable
- add more targeted tests for ambiguous project resolution
- tighten GUI automation logging and focus verification
- add GPU diagnostics to startup health reporting

## 4. Proposed Deliverables

### Code deliverables
- smaller and more testable pipeline stages
- explicit direct-file and project-query document helpers
- durable session-memory persistence
- configurable RAG/document policy thresholds
- better automation health surfaces
- startup diagnostics for GPU and RAG readiness

### Test deliverables
- project-resolution regression tests
- pipeline stage behavior tests
- session continuity tests
- automation focus/reliability tests
- GPU fallback and low-power-mode tests

### Product deliverables
- clearer runtime health reporting
- more natural conversational continuity
- safer desktop automation behavior
- a stable base for GUI and MCP integration

## 5. Success Criteria

The roadmap is working if:
- the turn pipeline is readable and modular
- project lookup is accurate and explainable
- session continuity survives restarts
- automation failures are visible and recoverable
- GPU readiness is explicit rather than assumed
- GUI/MCP integration can attach without re-architecting the runtime

## 6. Notes

This plan intentionally avoids adding more assistant features until the current runtime is easier to maintain. The project is already capable enough. The next step is to make that capability reliable, observable, and product-grade.
