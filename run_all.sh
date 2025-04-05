#!/bin/bash

mkdir -p output_logs

# echo "Starting training for n = 3"
# python3 run.py 3 train 2>&1 | tee output_logs/run_3.log

echo "Starting training for n = 9"
python3 run.py 9 train 2>&1 | tee output_logs/run_9.log

echo "Starting training for n = 18"
python3 run.py 18 train 2>&1 | tee output_logs/run_18.log

echo "All training sessions completed. Logs are stored in output_logs."
