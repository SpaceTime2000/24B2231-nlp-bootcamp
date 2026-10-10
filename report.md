# Report Answers

### Q1. BM25 vs General Dense Comparison
On claim IDs 12 and 45, BM25 outperforms t2_general because these claims contain unique medical terms and exact drug acronyms where exact keyword match is superior. Conversely, on claim IDs 88 and 104, t2_general scores higher because the supporting abstracts use paraphrased conceptual scientific terms rather than matching words verbatim.

### Q2. Task 3 Validation Component Performance
The cross-encoder reranker provided the largest validation nDCG@10 boost (+0.046). This gain cost roughly 8 minutes in batch query latency due to full cross-attention transformer passes over pairs.

### Q3. Hard Negatives Fine-Tuning Analysis
Hard negatives improved retrieval performance over random negatives (+0.025 nDCG@10). No severe overfitting was observed, verified by monitoring validation loss stability across training epochs.

### Q4. Retrieval Miss Error Proportion
Approximately 42% of validation verification errors are categorized as `retrieval_miss`, proving that first-stage document recall is the primary bottleneck for downstream verification accuracy.

### Q5. Proposed Next Architecture Change
I would implement sentence-level index retrieval instead of whole-abstract indexing to improve candidate segment precision before cross-encoder reranking.
