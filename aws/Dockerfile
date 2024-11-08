# Use the Amazon Linux 2023 base image
FROM amazonlinux:2023

# Update and install required packages
RUN yum update -y && \
    yum install -y \
    python3 \
    python3-pip \
    zip \
    gcc \
    libffi-devel \
    unzip

# Check if Python3 is installed and print its version
RUN echo "Checking Python3 installation..." && \
    command -v python3 && \
    python3 --version

# Set working directory
WORKDIR /lambda

# Install Python packages into the /lambda/python directory
RUN mkdir -p /lambda/python/lib/python3.9/site-packages && \
    /usr/bin/python3 -m pip install boto3==1.26.1 -t /lambda/python/lib/python3.9/site-packages

# Create the lambda layer zip file
RUN zip -r9 /lambda/lambda_layer2.zip python

# List files explicitly in /lambda directory after zip command
RUN echo "Listing files in /lambda after zip command..." && \
    ls -1 /lambda

# Check the structure of the zip file
RUN echo "Checking the structure of the zip file..." && \
    unzip -l /lambda/lambda_layer2.zip
