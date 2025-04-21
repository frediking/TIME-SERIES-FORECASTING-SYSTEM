# Service Level Agreements

## Performance Guarantees
| Metric | Target | Remediation |
|--------|--------|-------------|
| Uptime | 99.95% | Credit 5%/hr |
| Latency | <500ms | Priority fix |
| Accuracy | ±2% | Model refresh |

## Support Tiers
```python
class Support:
    priority: Literal['critical','high','normal']
    response_time: timedelta
    escalation_path: List[str]
```
