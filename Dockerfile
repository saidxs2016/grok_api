# Base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Expose the application port
EXPOSE 8091

# Run the Dash app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8091", "--reload"]
