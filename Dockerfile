FROM python:3.13-slim

WORKDIR /app

# 1. Install dependencies first (takes advantage of Docker's layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the code files needed for training, registration, and serving
COPY generate_data.py .
COPY train.py .
COPY register_model.py .
COPY serve.py .

# 3. Execute the pipeline to compile model artifacts inside Linux
# This generates native Linux paths (file:///app/mlartifacts/...) in mlflow.db
RUN python generate_data.py
RUN python train.py
RUN python register_model.py

# 4. Clean up raw training scripts to keep the final production image slim
RUN rm generate_data.py train.py register_model.py

EXPOSE 8000

# 5. Start the FastAPI serving layer
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]

# FROM python:3.13-slim starts from a lightweight Python image, keeping the final container small.

# COPY requirements.txt and RUN pip install install dependencies first. Docker caches this layer, so rebuilds are fast unless dependencies change.

# COPY serve.py ., COPY mlflow.db ., and COPY mlartifacts/ bundle your API code, model registry database, and actual model files into the image.

# CMD tells Docker to start uvicorn when the container launches, listening on all interfaces (0.0.0.0) so traffic from outside the container can reach it.