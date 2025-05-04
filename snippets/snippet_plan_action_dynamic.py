#written by Gemini
# Source: agent logic.txt
# Purpose: Dynamic planning with context and heuristics.

def plan_action_dynamic(agent_id, current_goal_id, current_goal_data, intent_tr, feature_tr, context_tr, analytics_tr):
    """
    Selects a feature dynamically using context, feature status, and success rate heuristic.
    """
    trace_id = current_goal_data.get("trace_id", "unknown_trace")
    print(f"\n--- [{agent_id}] Dynamic Planning for Goal: {current_goal_data['intent']} ({current_goal_id}) ---")
    analytics_tr.log_event(agent_id, "planning_start", trace_id, {"goal": current_goal_data})

    required_tags = current_goal_data.get("required_tags", [])
    if not required_tags:
        print("Goal has no required tags. Cannot plan.")
        analytics_tr.log_event(agent_id, "planning_failure", trace_id, {"reason": "No required tags in goal"})
        return None, None, "No required tags"

    # Get Context
    current_context_dict = context_tr.get_current_context_dict()
    context_tags_list = context_tr.get_context_tags()
    analytics_tr.log_event(agent_id, "context_query", trace_id, {"context_snapshot": current_context_dict})

    # Find Candidate Features
    candidate_features = feature_tr.find_features_by_tags(required_tags, status_filter="active")
    all_features_considered = list(feature_tr._data.keys())

    if not candidate_features:
        print("Selection: No suitable active features found.")
        selection_criteria = "No active features match required tags"
        selected_feature_id = None
    else:
        print(f"Candidate Features: {[f[0] for f in candidate_features]}")

        # Apply Heuristics
        preferred_feature = current_context_dict.get("user_preference_light")
        candidates_with_scores = []
        for feature_id, feature_data in candidate_features:
            score = feature_data.get("success_rate", 0.75)
            if feature_id == preferred_feature:
                score += 0.1
                print(f"  - Applying preference boost to {feature_id}")
            candidates_with_scores.append((score, feature_id, feature_data))

        candidates_with_scores.sort(key=lambda x: x[0], reverse=True)
        selected_score, selected_feature_id, selected_feature_data = candidates_with_scores[0]
        selection_criteria = f"Selected best candidate based on score (preference + success_rate): {selected_score:.2f}"
        print(f"Selection: Chose '{selected_feature_id}' based on '{selection_criteria}'")

    # Log Decision
    planning_details = {
        "goal": current_goal_data,
        "considered_features": all_features_considered,
        "candidate_features": [f[0] for f in candidate_features],
        "relevant_context_tags": context_tags_list,
        "selection_criteria": selection_criteria,
        "selected_feature": selected_feature_id
    }
    analytics_tr.log_event(agent_id, "planning_decision", trace_id, planning_details)

    if selected_feature_id:
        return selected_feature_id, selected_feature_data.get("slot"), selection_criteria
    else:
        return None, None, selection_criteria