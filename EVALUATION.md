# Evaluation & Benchmark Framework

## Benchmark Systems

1. **Baseline A**: Base Qwen2.5-3B-Instruct without RAG
2. **Baseline B**: Qwen2.5-3B + Vector-Only RAG
3. **System C**: Qwen2.5-3B + Hybrid GraphRAG
4. **System D**: Fine-tuned Qwen2.5-3B + Hybrid GraphRAG + Evidence Validator

## Metrics Tracked

- **Retrieval**: Recall@5, Recall@10, MRR, nDCG
- **Citations**: Citation Precision, Citation Recall / Completeness
- **Hallucination**: Unsupported Claim Rate
- **Abstention**: Abstention Accuracy, False Abstention Rate

## Running Evaluation

```bash
python scripts/run_evaluation.py --system all
```
Outputs comparison CSV and JSON artifacts to `data/evaluation/results/`.
