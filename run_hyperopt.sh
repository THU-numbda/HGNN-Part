#!/bin/bash

# GraphPart Hyperparameter Optimization Automation Script
# This script runs comprehensive hyperparameter optimization to achieve:
# - Cut loss < 0.02
# - Balance loss < 1e-3  
# - High-quality partitioning without post-processing

set -e  # Exit on any error

echo "🚀 GraphPart Hyperparameter Optimization"
echo "=========================================="

# Default configuration
DATA_PATH="./data"
OUTPUT_DIR="./hyperopt_results"
N_TRIALS=200
N_PARALLEL=4
TIMEOUT_HOURS=24

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --n-trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --n-parallel)
            N_PARALLEL="$2"
            shift 2
            ;;
        --timeout-hours)
            TIMEOUT_HOURS="$2"
            shift 2
            ;;
        --quick-test)
            N_TRIALS=20
            TIMEOUT_HOURS=2
            echo "🏃 Quick test mode: $N_TRIALS trials, $TIMEOUT_HOURS hour timeout"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --data-path PATH        Path to hypergraph data (default: ./data)"
            echo "  --output-dir PATH       Output directory (default: ./hyperopt_results)"
            echo "  --n-trials N           Number of optimization trials (default: 200)"
            echo "  --n-parallel N          Number of parallel workers (default: 4)"
            echo "  --timeout-hours N       Maximum runtime in hours (default: 24)"
            echo "  --quick-test           Run quick test with 20 trials and 2h timeout"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate inputs
if [ ! -d "$DATA_PATH" ]; then
    echo "❌ Error: Data path '$DATA_PATH' does not exist"
    exit 1
fi

if [ ! -f "$DATA_PATH"/*.hgr ]; then
    echo "❌ Error: No .hgr files found in '$DATA_PATH'"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/configs"
mkdir -p "$OUTPUT_DIR/models"

# Setup Python environment check
echo "🔍 Checking Python environment..."
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.7+"
    exit 1
fi

# Check required packages (basic check)
python -c "import torch, optuna, numpy, scipy" 2>/dev/null || {
    echo "❌ Missing required packages. Install with:"
    echo "   pip install torch torch-geometric optuna numpy scipy scikit-learn"
    exit 1
}

echo "✅ Environment check passed"

# Log system info
echo "📊 System Information:"
echo "  - Data path: $DATA_PATH"
echo "  - Output directory: $OUTPUT_DIR" 
echo "  - Trials: $N_TRIALS"
echo "  - Parallel workers: $N_PARALLEL"
echo "  - Timeout: ${TIMEOUT_HOURS}h"
echo "  - GPU available: $(python -c "import torch; print(torch.cuda.is_available())")"
echo "  - CPU cores: $(python -c "import multiprocessing; print(multiprocessing.cpu_count())")"

# Create optimization configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CONFIG_FILE="$OUTPUT_DIR/configs/optimization_config_$TIMESTAMP.json"

cat > "$CONFIG_FILE" << EOF
{
  "optimization": {
    "n_trials": $N_TRIALS,
    "n_parallel": $N_PARALLEL,
    "timeout_hours": $TIMEOUT_HOURS,
    "targets": {
      "cut_loss": 0.02,
      "balance_loss": 1e-3,
      "quality_threshold": 0.9
    }
  },
  "search_space": {
    "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
    "beta": {"type": "uniform", "low": 5.0, "high": 50.0},
    "gamma": {"type": "uniform", "low": 2.0, "high": 20.0},
    "alpha": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
    "dropout_rate": {"type": "uniform", "low": 0.0, "high": 0.5},
    "mask_rate": {"type": "uniform", "low": 0.1, "high": 0.4}
  },
  "timestamp": "$TIMESTAMP",
  "data_path": "$DATA_PATH",
  "output_dir": "$OUTPUT_DIR"
}
EOF

echo "📝 Configuration saved to: $CONFIG_FILE"

# Create monitoring script
MONITOR_SCRIPT="$OUTPUT_DIR/monitor_progress.py"
cat > "$MONITOR_SCRIPT" << 'EOF'
#!/usr/bin/env python3
import sys
import time
import sqlite3
import json
from pathlib import Path

def monitor_optuna_progress(db_path, target_trials):
    """Monitor Optuna optimization progress"""
    if not Path(db_path).exists():
        print("Database not found yet...")
        return False, 0, []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get completed trials
        cursor.execute("""
            SELECT trial_id, value, state, datetime_complete 
            FROM trials 
            WHERE state = 'COMPLETE' 
            ORDER BY value ASC
        """)
        
        completed = cursor.fetchall()
        
        # Get total trials
        cursor.execute("SELECT COUNT(*) FROM trials")
        total_trials = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"Progress: {len(completed)}/{target_trials} trials completed")
        
        if completed:
            best_trial = completed[0]
            print(f"Best so far: Trial {best_trial[0]} with objective {best_trial[1]:.6f}")
            
            # Show recent best trials
            recent_best = completed[:5]
            print("Top 5 trials:")
            for i, trial in enumerate(recent_best, 1):
                print(f"  {i}. Trial {trial[0]}: {trial[1]:.6f}")
        
        return len(completed) >= target_trials, len(completed), completed
        
    except Exception as e:
        print(f"Error accessing database: {e}")
        return False, 0, []

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "./optuna_study.db"
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    
    while True:
        completed, current, trials = monitor_optuna_progress(db_path, target)
        if completed:
            print("🎉 Optimization completed!")
            break
        time.sleep(30)  # Check every 30 seconds
EOF

chmod +x "$MONITOR_SCRIPT"

# Function to cleanup on exit
cleanup() {
    echo "🧹 Cleaning up..."
    # Kill any background processes
    jobs -p | xargs -r kill
    exit
}
trap cleanup EXIT INT TERM

# Start optimization with timeout
echo "🚀 Starting hyperparameter optimization..."
echo "⏰ Maximum runtime: ${TIMEOUT_HOURS} hours"

# Start the optimization process
timeout ${TIMEOUT_HOURS}h python hyperparameter_optimizer.py \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --n-trials $N_TRIALS \
    --n-parallel $N_PARALLEL &

OPTIM_PID=$!

# Start progress monitoring in background
sleep 10  # Give optimization time to start
python "$MONITOR_SCRIPT" "$OUTPUT_DIR/optuna_study.db" $N_TRIALS &
MONITOR_PID=$!

# Wait for optimization to complete
wait $OPTIM_PID
OPTIM_EXIT_CODE=$?

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true

# Check results
if [ $OPTIM_EXIT_CODE -eq 0 ]; then
    echo "✅ Optimization completed successfully!"
    
    # Check if we achieved targets
    if [ -f "$OUTPUT_DIR/best_config.json" ]; then
        echo "📊 Analyzing results..."
        python -c "
import json
try:
    with open('$OUTPUT_DIR/best_config.json') as f:
        results = json.load(f)
    
    print('\\n🎯 OPTIMIZATION RESULTS')
    print('=' * 50)
    
    if 'best_cut_loss' in results and results['best_cut_loss']:
        cut_loss = results['best_cut_loss'].get('cut_loss', float('inf'))
        print(f'Best Cut Loss: {cut_loss:.6f} (target: ≤ 0.02)')
        print('✅ Cut target achieved!' if cut_loss <= 0.02 else '❌ Cut target missed')
    
    if 'best_balance_loss' in results and results['best_balance_loss']:
        balance_loss = results['best_balance_loss'].get('balance_loss', float('inf'))
        print(f'Best Balance Loss: {balance_loss:.6f} (target: ≤ 1e-3)')
        print('✅ Balance target achieved!' if balance_loss <= 1e-3 else '❌ Balance target missed')
    
    print(f'\\n📁 Results saved to: $OUTPUT_DIR')
    print('📈 Check optuna-dashboard for detailed analysis:')
    print(f'   optuna-dashboard sqlite:///$OUTPUT_DIR/optuna_study.db')
    
except Exception as e:
    print(f'Error analyzing results: {e}')
    print('Raw results may be available in $OUTPUT_DIR/')
"
    else
        echo "⚠️  Best config file not found, but optimization completed"
    fi
    
elif [ $OPTIM_EXIT_CODE -eq 124 ]; then
    echo "⏰ Optimization timed out after ${TIMEOUT_HOURS} hours"
    echo "📊 Partial results may be available in $OUTPUT_DIR/"
else
    echo "❌ Optimization failed with exit code: $OPTIM_EXIT_CODE"
    echo "📋 Check logs in $OUTPUT_DIR/logs/ for details"
fi

# Generate summary report
echo "📝 Generating summary report..."
REPORT_FILE="$OUTPUT_DIR/optimization_summary_$TIMESTAMP.txt"

cat > "$REPORT_FILE" << EOF
GraphPart Hyperparameter Optimization Summary
Generated: $(date)
============================================

Configuration:
- Data path: $DATA_PATH
- Output directory: $OUTPUT_DIR
- Number of trials: $N_TRIALS
- Parallel workers: $N_PARALLEL
- Timeout: ${TIMEOUT_HOURS}h
- Exit code: $OPTIM_EXIT_CODE

Targets:
- Cut loss: ≤ 0.02
- Balance loss: ≤ 1e-3
- High partition quality without post-processing

Results:
$([ -f "$OUTPUT_DIR/best_config.json" ] && cat "$OUTPUT_DIR/best_config.json" | head -20 || echo "Results file not available")

Files generated:
$(ls -la "$OUTPUT_DIR/" | grep -E '\.(json|db|pt)$')

Next steps:
1. Check detailed results in $OUTPUT_DIR/
2. Use optuna-dashboard for interactive analysis:
   optuna-dashboard sqlite://$OUTPUT_DIR/optuna_study.db
3. Test best configuration with enhanced_trainer.py
4. Validate results on additional datasets

EOF

echo "📋 Summary report saved to: $REPORT_FILE"

# Final status
echo ""
echo "🏁 OPTIMIZATION COMPLETE"
echo "========================"
echo "📁 All results saved to: $OUTPUT_DIR"
echo "📊 View interactive dashboard:"
echo "   optuna-dashboard sqlite://$OUTPUT_DIR/optuna_study.db"
echo ""

# If successful, suggest next steps
if [ $OPTIM_EXIT_CODE -eq 0 ]; then
    echo "🚀 Next Steps:"
    echo "1. Review best configurations in $OUTPUT_DIR/best_config.json"
    echo "2. Run validation with: python enhanced_trainer.py --config $OUTPUT_DIR/best_config.json --data-path $DATA_PATH"
    echo "3. Deploy best model for inference"
fi

exit $OPTIM_EXIT_CODE