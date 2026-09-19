# Evaluation guide

Trajectory evaluation checks the sequence of actions an agent takes, including which tools it calls, the order of those calls, and whether it repeats unnecessary steps.

An LLM-as-a-judge can score an answer against a rubric, but judge scores should be calibrated with human-labeled examples because model-based judgments can be inconsistent or biased.
