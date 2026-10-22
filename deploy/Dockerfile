# Imagen del backend + frontend. Construir desde la raíz del repo:
#   docker build -f deploy/Dockerfile -t labdino .
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "httpx>=0.28" "numpy>=2" pandas matplotlib fastapi "uvicorn[standard]"
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .
COPY name_generator/artifacts ./name_generator/artifacts
COPY ollama/identity.json ./ollama/identity.json
COPY image_generator/outputs ./image_generator/outputs
COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
