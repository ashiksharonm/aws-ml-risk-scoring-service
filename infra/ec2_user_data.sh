#!/bin/bash
# Update and install dependencies
sudo yum update -y
sudo yum install -y git docker docker-buildx-plugin

# Start Docker service
sudo service docker start
sudo usermod -a -G docker ec2-user
sudo systemctl enable docker

# Install Docker Compose (optional but helpful)
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Log output
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "System Ready for Deployment"
