#!/bin/bash
set -e

# Configuration
STACK_NAME="risk-scoring-service"
REGION="us-east-1"
BUCKET_NAME="risk-scoring-artifacts-$(aws sts get-caller-identity --query Account --output text)"

echo "Deploying to AWS Lambda (Serverless Mode)..."

# 1. Ensure artifact bucket exists (SAM needs it for packaging code)
if aws s3 ls "s3://$BUCKET_NAME" 2>&1 | grep -q 'NoSuchBucket'; then
    echo "Creating artifact bucket: $BUCKET_NAME"
    aws s3 mb "s3://$BUCKET_NAME" --region "$REGION"
fi

# 2. Build the application
echo "Building..."
sam build --template infra/sam/template.yaml

# 3. Package and Deploy
echo "Deploying..."
sam deploy \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --capabilities CAPABILITY_IAM \
    --resolve-s3 \
    --confirm-changeset

echo "Deployment complete!"
