from pathlib import Path
import json
import time
import re

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


MODEL_NAME = "google/gemma-3-4b-it"
OUTPUT_PATH = Path(
    "results/gemma-task-planning/navigation_orchestration_v3.json"
)

MAX_NEW_TOKENS = 250
MAX_STEPS = 8

COMMAND = (
    "Find the red door and go there. "
    "If you can't reach it, try another way."
)

TOOLS = {
    "LOCALIZE": (
        "Determine the robot's current position and orientation."
    ),
    "QUERY_MAP": (
        "Query the semantic map for a named object or destination."
    ),
    "NAVIGATE_TO": (
        "Navigate the robot toward a known target."
    ),
    "ASK_VLM_ABOUT_CURRENT_VIEW": (
        "Ask the visual perception system what is visible "
        "in the robot's current camera view."
    ),
    "VERIFY_LOCATION": (
        "Verify whether the robot has reached the intended target."
    ),
}


def build_prompt(command, state, previous_result):
    previous_text = (
        json.dumps(previous_result, indent=2)
        if previous_result is not None
        else "None"
    )

    state_text = json.dumps(state, indent=2)

    tools_text = "\n".join(
        f"- {name}: {description}"
        for name, description in TOOLS.items()
    )

    return f"""
You are the high-level orchestrator for a mobile robot.

Your job is NOT to generate a complete plan in advance.

Instead, choose exactly ONE next action at a time based on:
1. The user's command.
2. The robot's current state.
3. The result of the previous tool call.

After a tool returns, you will be called again and must decide
what should happen next.

User command:
{command}

Available tools:
{tools_text}

Current robot state:
{state_text}

Previous tool result:
{previous_text}

Important rules:
- Choose exactly ONE tool for the next step.
- Do not output a complete multi-step plan.
- Use LOCALIZE when the robot's location is unknown.
- Use QUERY_MAP when map information could help.
- If the map does not contain the requested object, use
  ASK_VLM_ABOUT_CURRENT_VIEW to obtain visual information.
- Use NAVIGATE_TO when there is enough information to attempt
  navigation toward the target.
- If navigation fails, reconsider the situation and choose an
  appropriate next action.
- A failed route does not necessarily mean the target is unreachable.
- If another route may be available, attempt an alternative.
- Use VERIFY_LOCATION after navigation to confirm the goal.
- Do not invent tools.
- Do not assume that a tool succeeded when its result says it failed.
- Base your next decision on the actual tool result.

Return ONLY one JSON object in this format:

{{
  "action": "TOOL_NAME",
  "reason": "Brief explanation of why this is the next action",
  "target": "target name or null"
}}
""".strip()


def extract_json(text):
    text = text.strip()

    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(1)), False
        except json.JSONDecodeError:
            pass

    match = re.search(r"(\{.*\})", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(1)), False
        except json.JSONDecodeError:
            pass

    return None, False


def simulate_tool(action, target, state):
    if action == "LOCALIZE":
        return {
            "tool": "LOCALIZE",
            "status": "success",
            "robot_position": "office lobby",
            "orientation": "toward main corridor",
        }

    if action == "QUERY_MAP":
        if target == "red door":
            return {
                "tool": "QUERY_MAP",
                "status": "success",
                "target_found": False,
                "message": (
                    "No object named 'red door' is registered "
                    "in the semantic map."
                ),
            }

        return {
            "tool": "QUERY_MAP",
            "status": "success",
            "target_found": False,
            "message": "Target not found in the semantic map.",
        }

    if action == "ASK_VLM_ABOUT_CURRENT_VIEW":
        return {
            "tool": "ASK_VLM_ABOUT_CURRENT_VIEW",
            "status": "success",
            "observation": (
                "A red door is visible ahead on the right side "
                "of the corridor."
            ),
            "target": "red door",
            "relative_location": "ahead on the right",
        }

    if action == "NAVIGATE_TO":
        # ---------------------------------------------------------
        # FAILURE INJECTION:
        # The FIRST navigation attempt deliberately fails.
        # The SECOND attempt succeeds via an alternative route.
        # ---------------------------------------------------------
        if state["navigation_attempts"] == 0:
            state["navigation_attempts"] += 1

            return {
                "tool": "NAVIGATE_TO",
                "status": "failed",
                "target": target,
                "failure_reason": (
                    "The planned route is blocked by an "
                    "obstacle in the corridor."
                ),
                "alternative_route_available": True,
            }

        # Second navigation attempt succeeds.
        state["navigation_attempts"] += 1

        return {
            "tool": "NAVIGATE_TO",
            "status": "success",
            "target": target,
            "message": (
                "Navigation to 'red door' completed successfully "
                "using an alternative route."
            ),
        }
    if action == "VERIFY_LOCATION":
        return {
            "tool": "VERIFY_LOCATION",
            "status": "success",
            "verified": True,
            "message": (
                f"The robot has reached and verified "
                f"the target '{target}'."
            ),
        }

    return {
        "tool": action,
        "status": "error",
        "message": f"Unknown tool: {action}",
    }


def main():
    print(f"Loading model: {MODEL_NAME}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()

    state = {
        "robot_localized": False,
        "known_targets": [],
        "current_observation": None,
        "navigation_status": None,
        "navigation_attempts": 0,
        "verified": False,
    }

    previous_result = None
    trace = []

    total_start = time.perf_counter()

    for step in range(1, MAX_STEPS + 1):
        print()
        print("=" * 60)
        print(f"ORCHESTRATION STEP {step}")
        print("=" * 60)

        prompt = build_prompt(
            COMMAND,
            state,
            previous_result,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items()
            if hasattr(v, "to")
        }

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        latency = time.perf_counter() - start

        input_length = inputs["input_ids"].shape[-1]

        generated_tokens = generated[:, input_length:]

        answer = processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0].strip()

        peak_gpu_mb = None

        if torch.cuda.is_available():
            peak_gpu_mb = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 2)
            )

        print()
        print("Gemma:")
        print(answer)

        parsed, strict_json = extract_json(answer)

        if parsed is None:
            print()
            print("ERROR: Could not parse Gemma's response.")

            trace.append(
                {
                    "step": step,
                    "model_answer": answer,
                    "parsed_action": None,
                    "strict_json": strict_json,
                    "tool_result": None,
                    "state_after_tool": dict(state),
                    "latency_seconds": latency,
                    "peak_gpu_mb": peak_gpu_mb,
                }
            )

            break

        action = parsed.get("action")
        reason = parsed.get("reason")
        target = parsed.get("target")

        print()
        print(f"Action: {action}")
        print(f"Reason: {reason}")
        print(f"Target: {target}")

        if action not in TOOLS:
            print()
            print(f"ERROR: Unknown action: {action}")

            tool_result = simulate_tool(
                action,
                target,
                state,
            )

            trace.append(
                {
                    "step": step,
                    "model_answer": answer,
                    "parsed_action": parsed,
                    "strict_json": strict_json,
                    "tool_result": tool_result,
                    "state_after_tool": dict(state),
                    "latency_seconds": latency,
                    "peak_gpu_mb": peak_gpu_mb,
                }
            )

            break

        tool_result = simulate_tool(
            action,
            target,
            state,
        )

        print()
        print("Tool result:")
        print(json.dumps(tool_result, indent=2))

        # ---------------------------------------------------------
        # Update orchestration state from the actual tool result.
        # ---------------------------------------------------------
        if action == "LOCALIZE":
            if tool_result.get("status") == "success":
                state["robot_localized"] = True

        elif action == "QUERY_MAP":
            if tool_result.get("target_found"):
                target_name = target

                if (
                    target_name
                    and target_name not in state["known_targets"]
                ):
                    state["known_targets"].append(target_name)

        elif action == "ASK_VLM_ABOUT_CURRENT_VIEW":
            if tool_result.get("status") == "success":
                state["current_observation"] = (
                    tool_result.get("observation")
                )

                target_name = tool_result.get("target")

                if (
                    target_name
                    and target_name not in state["known_targets"]
                ):
                    state["known_targets"].append(target_name)

        elif action == "NAVIGATE_TO":
            state["navigation_status"] = tool_result.get(
                "status"
            )

        elif action == "VERIFY_LOCATION":
            state["verified"] = tool_result.get(
                "verified",
                False,
            )

        trace.append(
            {
                "step": step,
                "model_answer": answer,
                "parsed_action": parsed,
                "strict_json": strict_json,
                "tool_result": tool_result,
                "state_after_tool": dict(state),
                "latency_seconds": latency,
                "peak_gpu_mb": peak_gpu_mb,
            }
        )

        previous_result = tool_result

        if (
            action == "VERIFY_LOCATION"
            and tool_result.get("verified") is True
        ):
            print()
            print("Goal verified. Orchestration complete.")
            break

    total_latency = time.perf_counter() - total_start

    navigation_failures = sum(
        1
        for item in trace
        if (
            item["tool_result"] is not None
            and item["tool_result"].get("tool")
            == "NAVIGATE_TO"
            and item["tool_result"].get("status")
            == "failed"
        )
    )

    recovered_from_failure = (
        navigation_failures > 0
        and state["verified"] is True
    )

    result = {
        "model": MODEL_NAME,
        "experiment": "robot_orchestration_v2",
        "command": COMMAND,
        "architecture_role": (
            "closed-loop high-level robot orchestrator"
        ),
        "tools": TOOLS,
        "simulation": {
            "type": "simulated_tools",
            "failure_injection": (
                "First NAVIGATE_TO attempt fails because "
                "the planned route is blocked. "
                "A second attempt succeeds using an "
                "alternative route."
            ),
        },
        "evaluation": {
            "goal_verified": state["verified"],
            "navigation_failures": navigation_failures,
            "recovered_from_navigation_failure": (
                recovered_from_failure
            ),
            "total_steps": len(trace),
        },
        "trace": trace,
        "final_state": state,
        "total_latency_seconds": total_latency,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result: {OUTPUT_PATH}"
        )

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Steps executed: {len(trace)}")
    print(f"Navigation failures: {navigation_failures}")
    print(
        "Recovered from failure: "
        f"{recovered_from_failure}"
    )
    print(f"Goal verified: {state['verified']}")
    print(
        f"Total latency: {total_latency:.3f}s"
    )
    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()