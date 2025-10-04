# Use official Python image
FROM python:3.11

# Set working directory inside the container
WORKDIR /app

# Copy everything from your A3 folder into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Expose port (Dash runs on 80 for production)
EXPOSE 80

# Environment variables (will be passed from docker-compose)
ENV MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME}
ENV MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD}

# Run your Dash app
CMD ["python", "app.py"]
