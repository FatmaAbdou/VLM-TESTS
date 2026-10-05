from pathlib import Path
import json
import re
import time

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


MODEL_NAME = "google/gemma-3-4b-it"

OUTPUT_PATH = Path(
    "results/gemma-task-planning/"
    "navigation_orchestration_v1.json"
)

MAX_NEW_TOKENS = 250
MAX_STEPS = 8


COMMAND = "Find the red door and go there."


TOOLS = {
    "LOCALIZE": (
        "Determine the robot's current position and orientation."
    ),
    "QUERY_MAP": (
        "Query the robot's map for a named location or object."
    ),
    "NAVIGATE_TO": (
        "Send a navigation goal to Nav2 for a known target."
    ),
    "ASK_VLM_ABOUT_CURRENT_VIEW": (
        "Ask the vision-language model about what is visible "
        "in the robot's current camera view."
    ),
    "VERIFY_LOCATION": (
        "Verify that the robot has reached the intended target."
    ),
}


def build_prompt(command, state, tool_result):
    tools_text = "\n".join(
        f"- {name}: {description}"
        for name, description in TOOLS.items()
    )

    if tool_result is None:
        tool_result_text = "No tool has been called yet."
    else:
        tool_result_text = json.dumps(
            tool_result,
            indent=2,
        )

    return f"""
You are the high-level decision-making orchestrator
for a mobile robot.

Your job is to decide what the robot should do NEXT
in order to accomplish the user's command.

You are NOT being asked to produce the entire plan.

You must choose exactly ONE next action at each step.

The system will execute the selected action and return
its result to you. You will then decide what to do next.

User command:

"{command}"

Available tools:

{tools_text}

Current robot state:

{json.dumps(state, indent=2)}

Result from the previous tool call:

{tool_result_text}

Important rules:

1. Choose only ONE next action.
2. Base your decision on the current state and the
   result of the previous action.
3. Do not assume information that has not been provided.
4. Use the map when it can provide the required information.
5. Use visual perception when the required information
   cannot be obtained from the available map/state.
6. Do not invent tools or capabilities.
7. Do not execute the action yourself.
8. If the target location is known, navigation may be used.
9. After navigation, verify that the robot reached the target.
10. If the goal has already been achieved, select VERIFY_LOCATION
    or indicate that the task is complete.
11. Do not output a complete future plan.
12. Do not assume the result of a tool before it is called.

Return ONLY valid JSON.

Do NOT use Markdown.
Do NOT use ```json fences.
Do NOT include any text before or after the JSON.

Use exactly this structure:

{{
  "action": "TOOL_NAME",
  "reason": "Why this is the appropriate next action",
  "target": "Target if applicable"
}}

Valid actions are:

- LOCALIZE
- QUERY_MAP
- NAVIGATE_TO
- ASK_VLM_ABOUT_CURRENT_VIEW
- VERIFY_LOCATION

If the task is already complete, use:

{{
  "action": "VERIFY_LOCATION",
  "reason": "The target should now be verified.",
  "target": "..."
}}
"""


def extract_json(text):
    """
    Try strict JSON first.

    If the model incorrectly adds Markdown fences,
    extract the JSON object so the orchestration test
    can continue. We still record whether the original
    output was valid JSON.
    """

    try:
        return json.loads(text), True, None
    except json.JSONDecodeError as strict_error:
        pass

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL,
    )

    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed, False, None
        except json.JSONDecodeError as extracted_error:
            return None, False, str(extracted_error)

    return None, False, str(strict_error)


def simulate_tool(action, target, state):
    """
    Simulated robot tools.

    These are NOT real robot actions.
    They provide controlled results so we can evaluate
    Gemma's decision-making loop.
    """

    if action == "LOCALIZE":
        return {
            "tool": "LOCALIZE",
            "status": "success",
            "robot_position": "office lobby",
            "orientation": "toward main corridor",
        }

    if action == "QUERY_MAP":
        requested_target = target or "red door"

        if requested_target.lower() == "red door":
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
            "message": "Target was not found on the map.",
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
        if not target:
            return {
                "tool": "NAVIGATE_TO",
                "status": "failed",
                "message": (
                    "No navigation target was provided."
                ),
            }

        return {
            "tool": "NAVIGATE_TO",
            "status": "success",
            "target": target,
            "message": (
                f"Navigation to '{target}' completed successfully."
            ),
        }

    if action == "VERIFY_LOCATION":
        return {
            "tool": "VERIFY_LOCATION",
            "status": "success",
            "verified": True,
            "message": (
                f"The robot has reached and verified "
                f"the target '{target or 'red door'}'."
            ),
        }

    return {
        "tool": action,
        "status": "failed",
        "message": "Unknown tool.",
    }


def main():
    print(f"Loading model: {MODEL_NAME}")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        padding_side="left",
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model.eval()

    state = {
        "robot_localized": False,
        "known_targets": [],
        "current_observation": None,
        "navigation_status": None,
        "verified": False,
    }

    previous_result = None

    trace = []

    total_start = time.perf_counter()

    for step in range(1, MAX_STEPS + 1):
        print(f"\n{'=' * 60}")
        print(f"ORCHESTRATION STEP {step}")
        print(f"{'=' * 60}")

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
            key: value.to(model.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        latency = time.perf_counter() - start

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_length:]

        answer = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        peak_gpu_memory_mb = None

        if torch.cuda.is_available():
            peak_gpu_memory_mb = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 2)
            )

        parsed, strict_json, parse_error = extract_json(
            answer
        )

        print("\nGemma:")
        print(answer)

        if parsed is None:
            print("\nCould not parse Gemma's action.")
            print(f"Parse error: {parse_error}")

            trace.append(
                {
                    "step": step,
                    "model_answer": answer,
                    "parsed_action": None,
                    "strict_json": strict_json,
                    "parse_error": parse_error,
                    "latency_seconds": round(
                        latency,
                        3,
                    ),
                    "peak_gpu_memory_allocated_mb": (
                        round(
                            peak_gpu_memory_mb,
                            1,
                        )
                        if peak_gpu_memory_mb is not None
                        else None
                    ),
                }
            )

            break

        action = parsed.get("action")
        reason = parsed.get("reason")
        target = parsed.get("target")

        print(f"\nAction: {action}")
        print(f"Reason: {reason}")
        print(f"Target: {target}")

        if action not in TOOLS:
            print(
                f"\nERROR: Gemma selected an unavailable "
                f"action: {action}"
            )

            trace.append(
                {
                    "step": step,
                    "model_answer": answer,
                    "parsed_action": parsed,
                    "strict_json": strict_json,
                    "valid_action": False,
                    "parse_error": None,
                    "latency_seconds": round(
                        latency,
                        3,
                    ),
                    "peak_gpu_memory_allocated_mb": (
                        round(
                            peak_gpu_memory_mb,
                            1,
                        )
                        if peak_gpu_memory_mb is not None
                        else None
                    ),
                }
            )

            break

        tool_result = simulate_tool(
            action,
            target,
            state,
        )

        print("\nTool result:")
        print(
            json.dumps(
                tool_result,
                indent=2,
            )
        )

        # Update simulated world state.
        if action == "LOCALIZE":
            state["robot_localized"] = True

        elif action == "QUERY_MAP":
            if tool_result.get("target_found"):
                target_name = target or "unknown"
                state["known_targets"].append(
                    target_name
                )

        elif action == "ASK_VLM_ABOUT_CURRENT_VIEW":
            observation = tool_result.get(
                "observation"
            )

            state["current_observation"] = observation

            detected_target = tool_result.get(
                "target"
            )

            if detected_target:
                state["known_targets"].append(
                    detected_target
                )

        elif action == "NAVIGATE_TO":
            state["navigation_status"] = (
                tool_result.get("status")
            )

            if tool_result.get("status") == "success":
                state["current_observation"] = (
                    f"Robot is now at {target}."
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
                "valid_action": True,
                "tool_result": tool_result,
                "state_after_tool": dict(state),
                "latency_seconds": round(
                    latency,
                    3,
                ),
                "peak_gpu_memory_allocated_mb": (
                    round(
                        peak_gpu_memory_mb,
                        1,
                    )
                    if peak_gpu_memory_mb is not None
                    else None
                ),
            }
        )

        previous_result = tool_result

        if (
            action == "VERIFY_LOCATION"
            and tool_result.get("verified") is True
        ):
            print("\nGoal verified. Orchestration complete.")
            break

    total_latency = (
        time.perf_counter() - total_start
    )

    result = {
        "model": MODEL_NAME,
        "experiment": "robot_orchestration_v1",
        "command": COMMAND,
        "architecture_role": (
            "Gemma acts as a closed-loop high-level "
            "orchestrator. It selects one tool at a time, "
            "receives the tool result, updates its decision, "
            "and continues until the goal is verified."
        ),
        "tools": TOOLS,
        "simulation": {
            "enabled": True,
            "description": (
                "Tool calls are simulated. No real robot, "
                "Nav2, map service, or VLM is executed."
            ),
        },
        "trace": trace,
        "final_state": state,
        "total_latency_seconds": round(
            total_latency,
            3,
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result: "
            f"{OUTPUT_PATH}"
        )

    OUTPUT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n{'=' * 60}")
    print("FINAL RESULT")
    print(f"{'=' * 60}")
    print(
        f"Steps executed: {len(trace)}"
    )
    print(
        f"Total latency: {total_latency:.3f}s"
    )
    print(
        f"Goal verified: {state['verified']}"
    )
    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()