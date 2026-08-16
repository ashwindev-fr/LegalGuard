# Model Fine-Tuning Strategy

## Objectives

Fine-tuning is used strictly for **behavioral adaptation** (how to structure legal answers, cite evidence, disclose uncertainty, and abstain when evidence is missing), **NOT** for memorizing statutory knowledge (which is handled by RAG).

## Dataset Composition & Split

- **Split**: 70% Train, 15% Validation, 15% Test — split strictly by **document/case** to prevent chunk leakage across splits.
- **Method**: QLoRA / PEFT targeting attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
