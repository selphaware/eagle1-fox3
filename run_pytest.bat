@echo off
python -m pytest -ra -W default::Warning --no-cov tests
