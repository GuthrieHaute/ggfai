# DyTON-L: Conceptual Event Flow "Formula"

This document outlines the conceptual event flow and component interactions within the DyTON-L architecture. It represents the dynamic, event-driven nature of the system.

## Core Components & Entities:

* **`User`**: The external interactor.
* **`L-EBM`**: Lightweight Event Broker Middleware (e.g., Redis Pub/Sub, asyncio-based). Manages event publishing and subscription with prioritized queues and basic failover/retry.
* **`O-CR`**: Optimized Capabilities Registry (can be static/predefined or lightweight dynamic with caching). Stores information about node capabilities.
* **`OC`**: Orchestrator Core (with integrated Validation logic). Provides high-level coordination and policy enforcement.
* **Processing Nodes (Agents):**
    * **`IIN`**: Input Ingestion Node
    * **`CUN`**: ContextualUnderstandingNode (merged Context Aggregation & Understanding/Hypothesis Generation)
    * **`SPPN`**: Solution Planning & Proposal Node
    * **`UICN`**: User Interaction & Confirmation Node
    * **`EMN(s)`**: Execution & Monitoring Node(s)
    * **`LAN`**: Learning & Adaptation Node

**Event Payload Note:** All events (`EventName(...)`) are assumed to use compact binary formats (e.g., Protocol Buffers) for efficiency.

## Event Flow Sequence:

1.  **User Input & Initial Ingestion:**
    * `User` --(provides input)--> `IIN`
    * `IIN` --(sanitizes, identifies user)--> **`L-EBM`** : Publishes `NewRequestEvent(raw_input, user_id, request_id)`

2.  **Contextual Understanding:**
    * **`L-EBM`** --(delivers `NewRequestEvent`)--> `CUN`
    * `CUN` --(queries `O-CR` for context sources if needed; aggregates user history, profile, system state)--> `CUN`
    * `CUN` --(performs NLU: intent, entity, sentiment, ambiguity detection using lightweight models)--> `CUN`
    * `CUN` --(generates understanding hypotheses)--> **`L-EBM`** : Publishes `UnderstandingHypothesisEvent(s)(request_id, hypothesis_data, confidence_score)`

3.  **Solution Planning & Proposal:**
    * **`L-EBM`** --(delivers `UnderstandingHypothesisEvent(s)`)--> `SPPN`
    * `SPPN` --(evaluates hypotheses; queries `O-CR` for tools/actions based on hypothesis_data)--> `SPPN`
    * `SPPN` --(generates solution plan: actions, parameters, user summary)--> **`L-EBM`** : Publishes `ProposedSolutionPlanEvent(request_id, plan_data, user_summary_for_confirmation)`
        * *Alternative Path (if understanding is insufficient or direct clarification needed):*
            `SPPN` --> **`L-EBM`** : Publishes `ClarificationNeededEvent(request_id, question_for_user)`

4.  **Validation (by Orchestrator Core):**
    * **`L-EBM`** --(delivers `ProposedSolutionPlanEvent`)--> `OC`
    * `OC` --(validates `plan_data` against policies, permissions, resource constraints via `O-CR`)--> `OC`
    * **If Valid:** `OC` --> **`L-EBM`** : Publishes `ValidatedPlanEvent(request_id, plan_data, user_summary_for_confirmation)`
    * **If Invalid:** `OC` --> **`L-EBM`** : Publishes `PlanInvalidatedEvent(request_id, reason)`
        * *(`PlanInvalidatedEvent` might trigger a loop back to `SPPN` for replanning or directly lead to a `ClarificationNeededEvent` via `UICN`)*

5.  **User Interaction & Confirmation:**
    * **`L-EBM`** --(delivers `ValidatedPlanEvent` or `ClarificationNeededEvent` or `PlanInvalidatedEvent` to `UICN` via high-priority queue)--> `UICN`
    * `UICN` --(presents `user_summary_for_confirmation` or `question_for_user` or `reason` to `User`)--> `User`
    * `User` --(provides response: confirm, reject, correct, answer)--> `UICN`
    * `UICN` --(processes response)--> **`L-EBM`** : Publishes `UserDecisionEvent(request_id, decision_type, confirmed_plan_data_or_correction_data_or_answer)`
        * `decision_type`: e.g., `CONFIRMED`, `REJECTED`, `CORRECTION_PROVIDED`, `CLARIFICATION_ANSWERED`

6.  **Execution (Conditional on User Confirmation):**
    * **If** `UserDecisionEvent.decision_type == CONFIRMED`:
        * **`L-EBM`** --(delivers `UserDecisionEvent` with `confirmed_plan_data`)--> `EMN(s)`
        * `EMN(s)` --(parses `confirmed_plan_data`; queries `O-CR` for tool specifics; executes actions)--> `EMN(s)`
        * `EMN(s)` --(monitors execution)--> **`L-EBM`** : Publishes `ExecutionStatusEvent(request_id, step_id_if_workflow, status, result_data, error_info_if_any)`
            * `status`: e.g., `SUCCESS`, `FAILURE`, `IN_PROGRESS`

7.  **Learning & Adaptation (Continuous Background Process):**
    * `LAN` --(subscribes to relevant events from **`L-EBM`**: `UserDecisionEvent` [especially `CORRECTION_PROVIDED`], `ExecutionStatusEvent` [failures/successes], direct user feedback if available)--> `LAN`
    * `LAN` --(performs online learning with lightweight models; updates model parameters, heuristics, or knowledge used by `CUN`, `SPPN`, or contributes to `O-CR` knowledge)--> (Updates internal models/knowledge stores)

---

This Markdown representation should clearly illustrate the event-driven nature and the responsibilities of each component in the DyTON-L architecture.
