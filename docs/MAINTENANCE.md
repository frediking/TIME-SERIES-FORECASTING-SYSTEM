# System Maintenance

## Common Tasks
```bash
# Prune old models
python -m maintenance prune --older-than 30d

# Backup registry
python -m maintenance backup --output s3://backups
```

## Monitoring Checks
| Metric | Healthy Range |
|--------|---------------|
| CPU | <70% |
| Memory | <80% |
| Queue | <100 |
