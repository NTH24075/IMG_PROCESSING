"""Compatibility entry point. Evaluation uses run_eval and its test lock."""
from object_counting.metrics import counting_metrics as metrics
from run_eval import main

if __name__ == '__main__':
    main()
