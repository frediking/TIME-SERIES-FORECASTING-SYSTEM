# Deployment Guide

## Requirements
- Python 3.10+
- Redis (for caching)
- 4GB+ RAM

## Configuration
Set these environment variables:
```bash
export MODEL_REGISTRY_PATH=/path/to/models
export MONITORING_WINDOW_SIZE=1000
```

## Production Setup
1. **Docker**:
```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

2. **Kubernetes**:
```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  template:
    containers:
    - name: forecasting-agent
      image: your-registry/forecasting-agent:latest
      envFrom:
      - configMapRef:
          name: agent-config
```
