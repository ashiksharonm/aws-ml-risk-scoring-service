# Model card

Generated at train time as `models/model_card.json` (served at `/model-card`); this file is the committed summary.

- **Model**: XGBoost classifier (300 trees, depth 6, lr 0.05) in a scikit-learn pipeline. **Challenger**: logistic regression on the same preprocessing.
- **Intended use**: portfolio demonstration of explainable credit-default scoring. Not validated for real lending.
- **Data**: UCI "Default of Credit Card Clients" (30,000 Taiwanese cardholders, 2005), 80/20 stratified split, seed 42. About 22% positive class.
- **Holdout metrics** (reproduced with `make train`):

| Model | ROC-AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|
| XGBoost | 0.774 | 0.817 | 0.660 | 0.359 |
| Logistic regression | 0.708 | 0.808 | - | - |

- **Explanations**: TreeSHAP log-odds. Reason codes use the top positive contributors and never report `sex, age, marriage, education`.
- **Limitations**: one country and year, no out-of-time test; recall on defaults is low at the 0.5 threshold; probabilities are not recalibrated; protected attributes are still model inputs and no fairness or disparate-impact audit was run; the always-"no default" baseline already scores about 78% accuracy, so use AUC.
