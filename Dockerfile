# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application code and all necessary files into the container
COPY bangalore_app.py /app/
COPY best_xgb_model.pkl /app/
COPY scaler.pkl /app/
COPY feature_names.pkl /app/
COPY requirements.txt /app/

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000
EXPOSE 8000

# Run the Streamlit app on port 8000
CMD ["streamlit", "run", "bangalore_app.py", "--server.port=8000", "--server.address=0.0.0.0"]

