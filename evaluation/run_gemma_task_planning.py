from pathlib import Path
import json
import time

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


MODEL_NAME = "google/gemma-3-4b-it"

OUTPUT_PATH = Path(
    "results/gemma-task-planning/navigation_task_planning_v3.json"
)

MAX_NEW_TOKENS = 600


COMMAND = (
    "Take me to the reception desk and then head over to the elevator. "
    "If you can't make it to the elevator, just come back to reception "
    "and stay there."
)


CAPABILITIES = [
    "LOCALIZE",
    "QUERY_MAP",
    "PLAN_PATH",
    "NAV2_GOTO",
    "MONITOR_NAVIGATION",
    "VERIFY_LOCATION",
]


def build_prompt(command):
    capabilities_text = """
1. LOCALIZE
   Determine the robot's current position and orientation.

2. QUERY_MAP
   Query the robot's map for a named location or object.

3. PLAN_PATH
   Generate a navigable path from the robot's current
   position to a target position.

4. NAV2_GOTO
   Send a target navigation goal to Nav2.

5. MONITOR_NAVIGATION
   Monitor whether navigation is progressing,
   blocked, failed, or completed.

6. VERIFY_LOCATION
   Verify that the robot has reached the intended
   destination.
"""

    return f"""
You are the high-level task planner for a mobile robot.

Your job is to convert a natural-language robot command
into an ordered execution plan.

The plan will eventually be passed to a robot execution
system. Individual actions may call services such as a map
system or Nav2.

Robot command:

"{command}"

Available robot capabilities:

{capabilities_text}

The planner may use these capabilities when appropriate.

The planner must NOT invent capabilities that are not
listed above.

Create a complete plan for executing the command.

Important rules:

1. Break the command into meaningful executable steps.
2. Infer the intended sequence of actions from the command.
3. Put the steps in the correct dependency order.
4. Identify and handle conditional instructions when present.
5. Use the available robot capabilities when needed.
6. Do not invent capabilities that are not available.
7. Do not directly execute any action.
8. Distinguish planning actions from execution actions.
9. Verify that destinations are reached before continuing.
10. If a navigation attempt fails, represent the appropriate
    fallback behavior in the plan.
11. If the robot must return to a previous location, plan that
    navigation explicitly.
12. Do not assume the robot already knows destination coordinates.
13. Include the Nav2 handoff for every navigation goal.
14. Keep the plan concise but complete.

Return ONLY valid JSON.

Do NOT use Markdown.
Do NOT use ```json fences.
Do NOT include any text before or after the JSON.

Use exactly this structure:

{{
  "goal": "short description of the overall goal",
  "tasks": [
    {{
      "id": 1,
      "action": "ACTION_NAME",
      "description": "What this task does",
      "depends_on": []
    }}
  ]
}}

Valid action names are:

- LOCALIZE
- QUERY_MAP
- PLAN_PATH
- NAV2_GOTO
- MONITOR_NAVIGATION
- VERIFY_LOCATION
"""


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

    prompt = build_prompt(COMMAND)

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

    parsed_plan = None
    parse_error = None

    try:
        parsed_plan = json.loads(answer)
    except json.JSONDecodeError as exc:
        parse_error = str(exc)

    result = {
        "model": MODEL_NAME,
        "experiment": "robot_task_planning_v3",
        "command": COMMAND,
        "available_capabilities": CAPABILITIES,
        "prompt": prompt,
        "model_answer": answer,
        "parsed_plan": parsed_plan,
        "parse_error": parse_error,
        "latency_seconds": round(latency, 3),
        "peak_gpu_memory_allocated_mb": (
            round(peak_gpu_memory_mb, 1)
            if peak_gpu_memory_mb is not None
            else None
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result: {OUTPUT_PATH}"
        )

    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nModel answer:")
    print(answer)

    print(f"\nLatency: {latency:.3f}s")

    if peak_gpu_memory_mb is not None:
        print(
            f"Peak GPU memory: "
            f"{peak_gpu_memory_mb:.1f} MB"
        )

    print(f"\nSaved: {OUTPUT_PATH}")

    if parsed_plan is not None:
        print("JSON parsing: SUCCESS")
    else:
        print(
            f"JSON parsing: FAILED — {parse_error}"
        )


if __name__ == "__main__":
    main()