#!/bin/bash
# Run all tests with logging
python -m pytest test_*.py -v | tee test_results.log

# Verify hyperparameters loaded
grep -q "Loaded hyperparameters" agent/ensemble.py || echo "Hyperparameter loading failed"

# Check training output
if grep -q "ERROR" test_results.log; then
  echo "Tests failed"
  exit 1
else
  echo "All tests passed"
fi