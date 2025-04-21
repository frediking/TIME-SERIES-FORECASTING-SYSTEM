# Logging Architecture

## Log Types
```python
class LogEntry:
    timestamp: datetime
    level: Literal['DEBUG','INFO','WARN','ERROR']
    component: str
    message: str
    metrics: dict
```

## Retention Policy
| Log Level | Retention | Storage |
|-----------|-----------|---------|
| DEBUG | 7 days | Local |
| INFO | 30 days | S3 |
| ERROR | 1 year | S3 + DB |
