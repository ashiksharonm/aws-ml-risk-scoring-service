# AWS Free Tier Deployment Guide

This guide will help you deploy the **AWS ML Risk Scoring Service** to an AWS EC2 instance completely within the **Free Tier** limits.

## Prerequisites
- An AWS Account (Free Tier active).
- Terminal with SSH access.

## Step 1: Launch EC2 Instance

1.  **Login to AWS Console** and go to **EC2**.
2.  Click **Launch Instance**.
3.  **Name**: `ML-Risk-Service`.
4.  **AMI**: Select **Amazon Linux 2023 AMI** (Free tier eligible).
5.  **Instance Type**: Select **t2.micro** (or `t3.micro` if listed as Free Tier eligible in your region).
6.  **Key Pair**: Create a new key pair (e.g., `aws-ml-key`) and **download the .pem file**.
7.  **Network Settings**:
    - Allow SSH traffic from **My IP**.
    - Allow HTTP traffic from the internet (check the box).
    - Allow HTTPS traffic from the internet (check the box).
    - **IMPORTANT**: Click "Edit" and add a custom TCP rule for port **8000** (Source: `0.0.0.0/0` or `My IP`) if you want direct access, but Caddy will use 80/443.
8.  **Storage**: Default is fine (8GB gp3). You can go up to 30GB for free.
9.  **Advanced Details** (Optional):
    - Paste the content of `infra/ec2_user_data.sh` into the **User data** field to auto-install Docker.
10. Click **Launch Instance**.

## Step 2: Deploy Code

1.  **Get Public IP**: Go to the EC2 Dashboard and copy the **Public IPv4 address** of your new instance.
2.  **Transfer Code**:
    Open your terminal in the project directory and run:

    **Run this on your LOCAL MACHINE (New Terminal):**
    ```bash
    # Set permissions for your key
    chmod 400 path/to/aws-ml-key.pem

    # Copy files excluding heavy/unnecessary ones
    # Note: We need the 'models', 'src', 'docker', 'requirements.txt', plus config files
    scp -i path/to/aws-ml-key.pem -r src models docker requirements.txt docker-compose.yml Caddyfile ec2-user@<PUBLIC-IP>:/home/ec2-user/
    ```

    *Alternatively, if you pushed to GitHub, you can SSH in and `git clone`.*

3.  **Connect to Instance**:
    **Run this on your LOCAL MACHINE:**
    ```bash
    ssh -i path/to/aws-ml-key.pem ec2-user@<PUBLIC-IP>
    ```

## Step 3: Run Application

**Run these commands INSIDE the EC2 terminal:**

1.  **Set Domain & Email**:
    ```bash
    # Replace with your actual DuckDNS domain
    export DOMAIN_NAME=your-app.duckdns.org
    ```

    **Troubleshooting Build Error:**
    If you see `compose build requires buildx`, install the plugin:
    ```bash
    sudo yum install docker-buildx-plugin -y
    ```

2.  **Start Services**:
    ```bash
    cd /home/ec2-user

    # 1. Build the image manually (Reliable on all EC2 instances)
    docker build -t ml-api -f docker/Dockerfile .

    # 2. Start services using that image
    docker-compose up -d --no-build
    ```

3.  **View Logs** (Optional):
    ```bash
    docker-compose logs -f
    ```

## Step 4: Verify

1.  Open your browser to: `https://<YOUR-DOMAIN>/docs` (e.g., https://myapp.duckdns.org/docs)
2.  You should see the FastAPI Swagger UI with a secure lock icon 🔒.
3.  **Note**: It may take a minute for Caddy to fetch the SSL certificate on the first run.

## ⚠️ Important Billing Note
To stay free:
- **Stop or Terminate** the instance when you are done testing.
- Ensure you don't have other EC2 instances running that would exceed the 750 hours/month limit collectively.
