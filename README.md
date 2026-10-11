# AWS Financial Risk Prediction Engine

A production-grade Machine Learning service for credit risk scoring, deployed on AWS Free Tier.

## Architecture

<img width="1662" height="508" alt="image" src="https://github.com/user-attachments/assets/81faf79e-027f-4af0-b65d-1d8851a971a4" />


## Features

- **Model**: XGBoost classifier. `/predict` returns the top-5 per-feature contributions (log-odds) computed with XGBoost's native TreeSHAP (`pred_contribs=True`).
- **Dataset**: UCI Credit Default (Processed with Scikit-learn).
- **API**: FastAPI with clean Pydantic schemas.
- **Infrastructure**:
    - **Mode A**: Serverless (AWS SAM + Lambda).
    - **Mode B**: Dockerized (EC2 + Caddy + HTTPS).
- **DevOps**: CI/CD with GitHub Actions, Makefile for automation.

## Model Performance

Reproduced by running `make train` (UCI Default of Credit Card Clients, 30,000 rows, 80/20 stratified split, seed 42):

| Model | ROC-AUC | Accuracy | Precision | Recall |
|-------|---------|----------|-----------|--------|
| **XGBoost** | **0.774** | 0.817 | 0.660 | 0.359 |
| Logistic Regression (baseline) | 0.708 | 0.808 | - | - |

ROC-AUC is the headline metric: the classes are imbalanced (about 22% defaults), so a model that always predicts "no default" already scores about 77.9% accuracy, and the XGBoost accuracy of 81.7% is only modestly above that. Recall on defaults is low (0.359) at the default 0.5 threshold.

## Local Setup

1. **Clone & Install**
   ```bash
   make install
   ```
2. **Train Model**
   ```bash
   make train
   ```
   *Artifacts saved to `models/` and metric reports to `reports/`.*

3. **Run API**
   ```bash
   make run-api
   ```
   Open `http://localhost:8000/docs` to test endpoints.

## AWS Deployment

### Mode A: Serverless (Lambda)
Ensure you have `aws-cli` and `sam-cli` installed.
```bash
./infra/deploy_lambda.sh
```

### Mode B: EC2 (Docker) - **RECOMMENDED for Free Tier**
Follow the detailed [Deployment Guide](docs/DEPLOY_GUIDE.md).

**Key Features of this Deployment:**
- **Zero Cost**: Uses Free Tier `t2.micro` EC2 instance.
- **HTTPS Enabled**: Uses **Caddy** for automatic SSL termination (Let's Encrypt).
- **Dynamic DNS**: Integrates with **DuckDNS** for a free custom domain.
- **Containerized**: Orchestrated via `docker-compose` for stability.

**Quick Summary:**
1. Launch a `t2.micro` EC2 instance.
2. Transfer code via `scp`.
3. Set your DuckDNS domain: `export DOMAIN_NAME=yourapp.duckdns.org`
4. Run `docker-compose up -d` to start the secure stack.

## Monitoring & Logs
- **Lambda**: View CloudWatch Logs groups `/aws/lambda/RiskScoringFunction`.
- **EC2**: View local container logs via `docker-compose logs -f`.
- **API**: All requests return a `PredictionResponse` with latency measurements.

## Cost Management (AWS Free Tier)
- **Lambda**: 400,000 GB-seconds per month free. Use nominal memory (512MB).
- **EC2**: 750 hours/month of t2.micro or t3.micro.
- **S3**: 5GB storage free. Clean up artifacts periodically.
