# AWS ML Risk Scoring Service

A production-grade Machine Learning service for credit risk scoring, deployed on AWS Free Tier.

## Architecture

```ascii
                                   +-------------------+
                                   |   AWS API Gateway |
                                   +---------+---------+
                                             |
                                   +---------v---------+
                                   |    AWS Lambda     |
                                   | (FastAPI + Mangum)|
       Serverless Mode             +---------+---------+
                                             |
                                   +---------v---------+
                                   |      S3 Bucket    |
                                   |   (Model & Data)  |
                                   +-------------------+

       -----------------------------------------------------

                                   +-------------------+
                                   |   EC2 Instance    |
                                   |  +-------------+  |
       EC2 / Docker Mode           |  |    Nginx    |  |
                                   |  +------+------+  |
                                   |         |         |
                                   |  +------v------+  |
                                   |  |   FastAPI   |  |
                                   |  +-------------+  |
                                   +-------------------+
```

## Features

- **Model**: XGBoost Classifier with SHAP explainability.
- **Dataset**: UCI Credit Default (Processed with Scikit-learn).
- **API**: FastAPI with clean Pydantic schemas.
- **Infrastructure**:
    - **Mode A**: Serverless (AWS SAM + Lambda).
    - **Mode B**: Dockerized (EC2 + Nginx).
- **DevOps**: CI/CD with GitHub Actions, Makefile for automation.

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

### Mode B: EC2 (Docker)
1. Launch an EC2 instance (t2.micro).
2. Edit `infra/deploy_ec2.sh` with your key and IP.
3. Run:
   ```bash
   ./infra/deploy_ec2.sh
   ```

## Monitoring & Logs
- **Lambda**: View CloudWatch Logs groups `/aws/lambda/RiskScoringFunction`.
- **EC2**: View local container logs via `docker-compose logs -f`.
- **API**: All requests return a `PredictionResponse` with latency measurements.

## Cost Management (AWS Free Tier)
- **Lambda**: 400,000 GB-seconds per month free. Use nominal memory (512MB).
- **EC2**: 750 hours/month of t2.micro or t3.micro.
- **S3**: 5GB storage free. Clean up artifacts periodically.
