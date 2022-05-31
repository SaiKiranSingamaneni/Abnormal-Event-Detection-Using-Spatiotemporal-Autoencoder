#!/bin/bash

set -e

exec python3 train.py &
exec python3 test.py