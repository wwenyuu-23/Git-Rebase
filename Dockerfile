# Development environment for the EGT309 machine maintenance project.
# Runs JupyterLab with all project dependencies installed.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt jupyterlab

# Copy the project in (docker-compose mounts the live folder over this)
COPY . .

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token="]
