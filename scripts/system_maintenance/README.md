# System Maintenance Scripts

## Contents

1. `logging_config.py` - Centralized logging configuration:
   - Sets up JSON-formatted logs
   - Configures log levels and handlers
   - Used by all components via import

2. `run_tests.py` - Test execution:
   - Runs unit and integration tests
   - Generates coverage reports
   - Can be scheduled for CI/CD pipelines

## Usage
All project components should import logging via:
```python
from scripts.system_maintenance.logging_config import setup_logging
