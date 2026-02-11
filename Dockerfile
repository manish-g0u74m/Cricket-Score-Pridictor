FROM python:3.9-slim
WORKDIR /app
COPY src/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY models/ models/                   # this is not right to copy model in dockerfile
COPY src/main.py .
COPY src/templates templates
EXPOSE 5000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
