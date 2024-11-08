# Use a base image with a compatible framework
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1-cpu-py3

# Install additional dependencies if necessary
RUN pip install --upgrade pip
RUN pip install pandas scikit-learn

# Set the working directory inside the container
WORKDIR /opt/ml/code

# Copy your training script into the container
COPY train_female.py /opt/ml/code/train_female.py
COPY train_male.py /opt/ml/code/train_male.py

# Set the entry point for the container
ENTRYPOINT ["python", "/opt/ml/code/train.py"]
