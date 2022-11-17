## Predictions
This directory contains the predictions of our models. Additionally, the evaluation scores for the predictions are listed in the following table:

| Model                                       | Score   |
| ------------------------------------------- | ------- |
| SVD Baseline                                | 1.00324 |
| NMF Baseline                                | 1.00036 |
| SVD++ Baseline                              | 0.99881 |
| SlopeOne Baseline                           | 0.99832 |
| NCF Baseline                                | 1.02400 |
| GNN Baseline                                | 1.00004 |
| GNN with NCF                                | 0.98849 |
| Reinforced GNN with NCF (SlopeOne)          | 0.98544 |
| Ensemble Reinforced GNN with NCF (SlopeOne) | 0.98210 |

For the sake of completeness, we include a table with the test scores of our experiments used to determine the best combination of reinforcements:

| Model                                               | Score   |
| --------------------------------------------------- | ------- |
| Reinforced GNN with NCF (NMF, SlopeOne, SVD++)      | 0.98647 |
| Reinforced GNN with NCF (NMF, SlopeOne)             | 0.98695 |
| Reinforced GNN with NCF (NMF, SVD++)                | 0.98605 |
| Reinforced GNN with NCF (NMF)                       | 0.98679 |
| Reinforced GNN with NCF (SlopeOne, SVD++)           | 0.98689 |
| Reinforced GNN with NCF (SlopeOne)                  | 0.98544 |
| Reinforced GNN with NCF (SVD, NMF, SlopeOne, SVD++) | 0.98590 |
| Reinforced GNN with NCF (SVD, NMF, SlopeOne)        | 0.98703 |
| Reinforced GNN with NCF (SVD, NMF, SVD++)           | 0.98680 |
| Reinforced GNN with NCF (SVD, NMF)                  | 0.98801 |
| Reinforced GNN with NCF (SVD, SlopeOne, SVD++)      | 0.98637 |
| Reinforced GNN with NCF (SVD, SlopeOne)             | 0.98697 |
| Reinforced GNN with NCF (SVD, SVD++)                | 0.98749 |
| Reinforced GNN with NCF (SVD)                       | 0.98750 |
| Reinforced GNN with NCF (SVD++)                     | 0.98624 |
