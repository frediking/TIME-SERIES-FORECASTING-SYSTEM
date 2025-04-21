# CI/CD Pipeline

## Workflow Stages
```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: pytest --cov
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - run: ./deploy.sh
```

## Quality Gates
| Stage | Requirement |
|-------|-------------|
| Test | 90% coverage |
| Lint | 0 warnings |
| Build | <5 min |
