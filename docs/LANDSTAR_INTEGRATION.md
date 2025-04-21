# Landstar System Integration

## Data Sources
```mermaid
flowchart LR
    A[ELD Telematics] --> B
    C[McLeod TMS] --> B
    D[Fuel Cards] --> B
    B[Forecasting Agent]
```

## API Endpoints
| Service | Endpoint | Sample Use |
|---------|----------|------------|
| Capacity | `/v1/capacity/forecast` | Driver allocation |
| Pricing | `/v1/pricing/spot` | Bid guidance |
| Routing | `/v1/routing/optimize` | Load planning |
