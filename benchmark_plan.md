# VLM Evaluation Benchmark Plan

## 1. Evaluation Objective

The goal of this benchmark is to evaluate Vision-Language Models (VLMs) on a broad set of visual understanding capabilities relevant to real-world visual reasoning and navigation.

The benchmark is designed to be:

- Model-agnostic
- Dataset-agnostic
- Reproducible
- Ground-truth based
- Suitable for comparing multiple VLMs
- Capable of evaluating both static images and temporal/video inputs
- Focused on measurable visual capabilities rather than dataset-specific question types

The evaluation framework defines **WHAT capability is being tested**.

Datasets provide the:

- Images
- Videos
- Frames
- Questions
- Instructions
- Ground truth
- Reference annotations

A dataset must not redefine the benchmark's evaluation categories.

---

# 2. Evaluation Capabilities

The benchmark uses the following 16 capabilities.

## 2.1 Object Recognition

Identify objects or object classes visible in an image.

Examples:

- Is there a chair?
- What objects are visible?
- Is the object a bicycle or a motorcycle?

---

## 2.2 Dynamic Object Detection

Identify objects that are moving or changing position/state across frames.

Examples:

- Which object is moving?
- What object moved between the two frames?
- Which object changed position?

This capability is distinct from general temporal consistency.

---

## 2.3 Attribute Recognition

Identify visual attributes of objects.

Attributes may include:

- Color
- Shape
- Size
- Material
- Texture
- Appearance

Examples:

- What color is the object?
- Is the object large or small?
- What shape is the moving object?

---

## 2.4 Counting

Determine the number of visible objects or occurrences.

Examples:

- How many chairs are visible?
- How many objects are red?
- How many people are present?

Primary metrics:

- Counting accuracy
- Counting Mean Absolute Error (MAE)

---

## 2.5 Spatial Reasoning

Determine spatial relationships between objects.

Examples:

- What is to the left of the chair?
- Is the table behind the sofa?
- Which object is closest to the door?

Possible relations:

- Left / right
- Above / below
- In front / behind
- Near / far
- Inside / outside
- Between
- Relative position

---

## 2.6 Scene Understanding

Understand the overall environment and its context.

Examples:

- What type of environment is shown?
- What is the main activity?
- What type of room is visible?
- Which area appears to be a reception area?

Scene understanding should involve more than identifying an individual object.

---

## 2.7 Landmark Recognition

Identify recognizable landmarks or locations.

Examples:

- What landmark is shown?
- Which building/site is visible?
- Identify the depicted landmark.

This capability requires reliable ground-truth landmark identity.

---

## 2.8 Grounding / Instance Selection

Select or identify the specific object referred to by an expression.

Examples:

- Which object does "the red chair" refer to?
- Select the object described by the instruction.
- Which of the visible objects is being referred to?

This capability focuses on linking language to a specific visual instance.

---

## 2.9 Visibility / Occlusion

Determine whether an object is visible, partially visible, hidden, or occluded.

Examples:

- Is the object visible?
- Is the object partially blocked?
- Which object is behind the obstruction?
- Can the referenced object be seen?

---

## 2.10 Negative Recognition

Determine when an object or property is absent.

Examples:

- Is there a car in the image?
- Is there an animal?
- Are there any bicycles?
- Is there a red object?

Negative questions are important for measuring hallucination.

---

## 2.11 Conditional Reasoning

Answer questions requiring multiple visual conditions or logical relationships.

Examples:

- If the object is red, what shape is it?
- Which object satisfies both conditions?
- Is there an object that is both large and blue?

Conditional reasoning may involve multiple reasoning steps.

---

## 2.12 Instruction Following

Follow a natural-language visual instruction and produce the requested response.

Examples:

- Identify the object described by the instruction.
- Answer according to the requested format.
- Follow a visual task instruction.
- Determine what action/location is requested.

Instruction-following evaluation should measure both correctness and compliance with the requested output.

---

## 2.13 Obstacle / Navigation

Understand obstacles, navigability, and navigation-relevant spatial information.

Examples:

- Is the path blocked?
- What obstacle is in the way?
- Which direction should the agent move?
- Is the area accessible?
- What object prevents movement?

This capability is especially relevant to first-person navigation datasets.

---

## 2.14 Temporal Consistency

Maintain a consistent interpretation of objects, attributes, and relationships across multiple frames.

Examples:

- Is the same object present in both frames?
- Did the object's identity remain the same?
- Does the model maintain the same interpretation across frames?

This is different from simply detecting that something changed.

---

## 2.15 Change Detection

Identify what changed between two or more frames.

Examples:

- What changed between the two images?
- Which object moved?
- Did the object's color change?
- What appeared or disappeared?

---

## 2.16 Structured Output

Produce an answer in a required machine-readable or structured format.

Examples:

```json
{
  "object": "chair",
  "count": 3
}