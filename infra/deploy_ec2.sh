#!/bin/bash

# Configuration
KEY_PATH="path/to/your-key.pem"
EC2_USER="ec2-user"
EC2_HOST="your-ec2-public-ip"

echo "Deploying to EC2 instance..."
echo "Note: Ensure you have updated the variables in this script!"

# 1. Zip the project (excluding venv, git, etc)
echo "Packaging project..."
zip -r risk-scoring-service.zip . -x "*.git*" "venv/*" "__pycache__/*" "data/*" "models/*"
# Note: We exclude models/data if we want to run training on remote, but ideally we ship artifacts.
# If artifacts are large, use S3. For small, include them or re-train.
# Let's assume we ship artifacts for this demo.

# 2. SCP to EC2
# echo "Uploading..."
# scp -i $KEY_PATH risk-scoring-service.zip $EC2_USER@$EC2_HOST:/home/$EC2_USER/

# 3. SSH and Run
# echo "Connecting and deploying..."
# ssh -i $KEY_PATH $EC2_USER@$EC2_HOST << EOF
#     # Update system
#     sudo yum update -y
#     sudo amazon-linux-extras install docker
#     sudo service docker start
#     sudo usermod -a -G docker ec2-user
#     sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
#     sudo chmod +x /usr/local/bin/docker-compose
#
#     # Setup App
#     unzip -o risk-scoring-service.zip -d risk-scoring-service
#     cd risk-scoring-service
#     
#     # Build and Run
#     docker-compose -f docker/docker-compose.yml up -d --build
# EOF

echo "Deployment script template created. Uncomment commands and set variables to execute."
