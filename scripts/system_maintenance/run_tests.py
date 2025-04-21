#!/usr/bin/env python
"""
Test runner for the forecasting system.
Executes all tests and generates a coverage report.
"""
import os
import sys
import pytest
import argparse
from pathlib import Path

def run_tests(args):
    """Run tests with the specified options."""
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    # Build pytest arguments
    pytest_args = []
    
    # Add test directory or specific test files
    if args.test_files:
        pytest_args.extend(args.test_files)
    else:
        pytest_args.append(str(Path(__file__).parent))
    
    # Add verbosity
    if args.verbose:
        pytest_args.append('-v')
    
    # Add coverage
    if args.coverage:
        pytest_args.append('--cov=agent')
        pytest_args.append('--cov=config')
        pytest_args.append('--cov=utils')
        pytest_args.append('--cov=monitoring')
        
        if args.coverage_report:
            pytest_args.append(f'--cov-report={args.coverage_report}')
        else:
            pytest_args.append('--cov-report=term')
    
    # Add JUnit XML report
    if args.junit_xml:
        pytest_args.append(f'--junitxml={args.junit_xml}')
    
    # Run pytest with the built arguments
    return pytest.main(pytest_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tests for the forecasting system')
    parser.add_argument('test_files', nargs='*', help='Specific test files to run')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-c', '--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--coverage-report', choices=['term', 'html', 'xml'], help='Coverage report format')
    parser.add_argument('--junit-xml', help='Generate JUnit XML report')
    
    args = parser.parse_args()
    
    # Run tests
    exit_code = run_tests(args)
    
    # Exit with the same code as pytest
    sys.exit(exit_code)
