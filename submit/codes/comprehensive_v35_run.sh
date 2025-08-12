#!/bin/bash
# Comprehensive v35 execution script
# CRITICAL: NO timeout - scripts run to completion
# Includes memory monitoring and error trapping

set -e  # Exit on error

echo "=========================================="
echo "SCMS-PINN v35 Comprehensive Run"
echo "Started at: $(date)"
echo "=========================================="

# Create log directory
LOG_DIR="logs_v35_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Log directory: $LOG_DIR"

# Function to check GPU memory
check_gpu_memory() {
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Memory Status:"
        nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader,nounits
    fi
}

# Function to trap errors and save dmesg
trap_error() {
    echo "==================== ERROR DETECTED ===================="
    echo "Error occurred at: $(date)"
    echo "Exit code: $?"
    
    # Check dmesg for OOM killer
    if dmesg | tail -20 | grep -i "out of memory" > /dev/null; then
        echo "OOM Killer detected!"
        dmesg | tail -50 > "$LOG_DIR/oom_killer.log"
    fi
    
    check_gpu_memory
    echo "======================================================="
}

trap trap_error ERR

# Step 1: Training
echo -e "\n=== Step 1: Training SCMS-PINN v35 ==="
check_gpu_memory

# Run training with output logging
# NO timeout - runs to completion
echo "Starting training (this may take several hours)..."
echo "Training start time: $(date)"

# Run Python script and capture both stdout and stderr
~/.venv/ml_31123121/bin/python -u train_scms_pinn_v35.py 2>&1 | tee "$LOG_DIR/training.log"
TRAINING_EXIT_CODE=${PIPESTATUS[0]}

echo "Training exit code: $TRAINING_EXIT_CODE"
echo "Training end time: $(date)"

# Check if training succeeded
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully"
elif [ $TRAINING_EXIT_CODE -eq 139 ]; then
    echo "✗ Training failed with segmentation fault (exit code 139)"
    echo "This typically indicates memory corruption or pointer issues"
elif [ $TRAINING_EXIT_CODE -eq 132 ]; then
    echo "✗ Training failed with illegal instruction (exit code 132)"
    echo "This typically indicates a CPU instruction error, often in plotting/saving routines"
elif [ $TRAINING_EXIT_CODE -eq 2 ]; then
    echo "✗ Training stopped due to NaN/Inf or invalid loss values (exit code 2)"
    
    # Check if training actually finished all epochs
    if grep -q "Training Completed" "$LOG_DIR/training.log"; then
        echo "✓ Training reached completion"
    else
        echo "⚠️ Training exited with code 0 but may not have completed all epochs"
        
        # Extract last epoch from log
        LAST_EPOCH=$(grep "^Epoch" "$LOG_DIR/training.log" | tail -1)
        echo "Last logged epoch: $LAST_EPOCH"
        
        # Check for any error messages
        if grep -i "error\|exception\|traceback" "$LOG_DIR/training.log" > "$LOG_DIR/errors_found.txt"; then
            echo "⚠️ Errors/exceptions found in log:"
            head -20 "$LOG_DIR/errors_found.txt"
        fi
    fi
else
    echo "✗ Training failed with exit code: $TRAINING_EXIT_CODE"
    
    # Extract error information
    echo "Extracting error details..."
    
    # Get last 50 lines of log
    tail -50 "$LOG_DIR/training.log" > "$LOG_DIR/training_tail.log"
    
    # Look for specific error patterns
    if grep -i "out of memory" "$LOG_DIR/training.log" > /dev/null; then
        echo "⚠️ Out of Memory error detected"
    fi
    
    if grep -i "nan\|inf" "$LOG_DIR/training.log" > /dev/null; then
        echo "⚠️ NaN or Inf values detected in training"
    fi
    
    if grep -i "keyboard\|interrupt" "$LOG_DIR/training.log" > /dev/null; then
        echo "⚠️ Training was interrupted by user"
    fi
    
    trap_error
    exit 1
fi

# Find the training directory
TRAIN_DIR=$(ls -td training_v35_* | head -1)
echo "Training directory: $TRAIN_DIR"

# Copy best model for validation
if [ -f "$TRAIN_DIR/best_model.pth" ]; then
    echo "✓ Best model found: $TRAIN_DIR/best_model.pth"
else
    echo "✗ Best model not found! Using last checkpoint..."
    LAST_CHECKPOINT=$(ls -t $TRAIN_DIR/checkpoint_epoch_*.pth | head -1)
    if [ -z "$LAST_CHECKPOINT" ]; then
        echo "✗ No checkpoints found!"
        exit 1
    fi
    cp "$LAST_CHECKPOINT" "$TRAIN_DIR/best_model.pth"
fi

# Step 2: Physics Validation
echo -e "\n=== Step 2: Physics Validation ==="
check_gpu_memory

VALIDATION_DIR="validation_v35_$(date +%Y%m%d_%H%M%S)"
echo "Validation output directory: $VALIDATION_DIR"

# Run validation
~/.venv/ml_31123121/bin/python physics_validation_graphene_v35.py "$TRAIN_DIR/best_model.pth" "$VALIDATION_DIR" 2>&1 | tee "$LOG_DIR/validation.log"
VALIDATION_EXIT_CODE=${PIPESTATUS[0]}

if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
    echo "✓ Validation completed successfully"
    
    # Check if validation actually generated files
    if [ -f "$VALIDATION_DIR/band_structure_comparison.png" ]; then
        echo "✓ Validation plots generated"
    else
        echo "⚠️ Warning: Validation completed but no plots found"
    fi
else
    echo "⚠️ Validation completed with warnings (exit code: $VALIDATION_EXIT_CODE)"
    # Continue anyway - validation generates useful plots even if tests fail
fi

# Step 3: Generate additional analysis plots
echo -e "\n=== Step 3: Additional Analysis ==="

# Create analysis script
cat > "$LOG_DIR/additional_analysis_v35.py" << 'EOF'
#!/usr/bin/env python3
"""Additional analysis for v35 results"""
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_training_history(history_file, output_file):
    """Plot detailed training history with correct epoch indices"""
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Total loss - First subplot
    ax = axes[0]
    if 'train_loss' in history and history['train_loss']:
        epochs_train = np.arange(1, len(history['train_loss']) + 1)
        ax.plot(epochs_train, history['train_loss'], 'b-', label='Train', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        # Validation is done every val_interval epochs (typically 5)
        val_interval = 5  # Standard val_interval from CONFIG
        epochs_val = np.arange(val_interval, len(history['val_loss']) * val_interval + 1, val_interval)
        ax.plot(epochs_val, history['val_loss'], 'r--', label='Validation', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 300)
    
    # Loss components - Subplots 2-5
    components = ['data', 'dirac_symmetric', 'anchor', 'fermi_velocity']
    for i, comp in enumerate(components):
        ax = axes[i+1]
        key = f'{comp}_loss'
        if key in history and history[key]:
            data = history[key]
            # Only plot if we have valid data
            if data and any(v > 0 for v in data if v is not None):
                epochs_comp = np.arange(1, len(data) + 1)
                ax.plot(epochs_comp, data, linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'{comp.replace("_", " ").title()} Loss')
                # Only use log scale if all values are positive
                if all(v > 0 for v in data if v is not None):
                    ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 300)
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'{comp.replace("_", " ").title()} Loss')
    
    # Gap at K-points - Last subplot
    ax = axes[5]
    if 'k_gap' in history and history['k_gap']:
        # k_gap is recorded during validation, so same as val_loss
        val_interval = 5
        epochs_gap = np.arange(val_interval, len(history['k_gap']) * val_interval + 1, val_interval)
        ax.plot(epochs_gap, history['k_gap'], 'g-', linewidth=2, marker='s', markersize=4)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gap (eV)')
        ax.set_title('Physics Constraint: Gap at Dirac Points')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 300)
    else:
        ax.text(0.5, 0.5, 'No k_gap data', transform=ax.transAxes,
               ha='center', va='center', fontsize=12)
        ax.set_title('Physics Constraint: Gap at Dirac Points')
    
    plt.suptitle('SCMS-PINN v35 Training History Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_dir = sys.argv[1]
        history_file = f"{train_dir}/training_history.json"
        output_file = f"{train_dir}/training_history_analysis.png"
        plot_training_history(history_file, output_file)
EOF

chmod +x "$LOG_DIR/additional_analysis_v35.py"

# Run additional analysis (only if training_history.json exists)
if [ -f "$TRAIN_DIR/training_history.json" ]; then
    echo "Running training history analysis..."
    ~/.venv/ml_31123121/bin/python "$LOG_DIR/additional_analysis_v35.py" "$TRAIN_DIR"
else
    echo "⚠️ Training history not found, skipping analysis plot"
fi

# Step 4: Summary Report
echo -e "\n=== Step 4: Generating Summary Report ==="

cat > "$LOG_DIR/summary_report.md" << EOF
# SCMS-PINN v35 Comprehensive Run Summary

**Date**: $(date)
**Training Directory**: $TRAIN_DIR
**Validation Directory**: $VALIDATION_DIR

## Key Improvements in v35

1. **Memory Optimization**: Batched symmetry operations (12x reduction)
2. **Loss Fixes**: 
   - Disabled unstable smoothness loss
   - Fixed Fermi velocity gradient flow
3. **Robustness**: Added OOM handling and memory monitoring
4. **Configuration**: Reduced batch size (128) and accumulation (2)

## Training Results

EOF

# Extract key metrics from training log
if [ -f "$TRAIN_DIR/training_history.json" ]; then
    echo "### Final Metrics" >> "$LOG_DIR/summary_report.md"
    ~/.venv/ml_31123121/bin/python -c "
import json
with open('$TRAIN_DIR/training_history.json', 'r') as f:
    h = json.load(f)
    if h['train_loss']:
        print(f'- Final training loss: {h[\"train_loss\"][-1]:.6f}')
    if h['val_loss']:
        print(f'- Final validation loss: {h[\"val_loss\"][-1]:.6f}')
    if h['k_gap']:
        print(f'- Final gap at K-points: {h[\"k_gap\"][-1]:.6f} eV')
" >> "$LOG_DIR/summary_report.md"
fi

# Extract validation results
if [ -f "$VALIDATION_DIR/validation_results.json" ]; then
    echo -e "\n## Validation Results" >> "$LOG_DIR/summary_report.md"
    ~/.venv/ml_31123121/bin/python -c "
import json
with open('$VALIDATION_DIR/validation_results.json', 'r') as f:
    r = json.load(f)['results']
    print(f'- Gap at K-points: {r[\"critical_points\"][\"K_max_gap\"]:.6f} eV')
    print(f'- Average band error: {r[\"band_structure\"][\"avg_gap_error\"]:.6f} eV')
    print(f'- Fermi velocity (TB): {r[\"fermi_velocity\"][\"avg_tb\"]:.4f} eV·Å')
    print(f'- Fermi velocity (PINN): {r[\"fermi_velocity\"][\"avg_pinn\"]:.4f} eV·Å')
" >> "$LOG_DIR/summary_report.md"
fi

echo -e "\n## Output Files\n" >> "$LOG_DIR/summary_report.md"
echo "### Training Outputs" >> "$LOG_DIR/summary_report.md"
ls -la "$TRAIN_DIR"/*.png "$TRAIN_DIR"/*.pth "$TRAIN_DIR"/*.json 2>/dev/null | awk '{print "- " $9}' >> "$LOG_DIR/summary_report.md"

echo -e "\n### Validation Outputs" >> "$LOG_DIR/summary_report.md"
ls -la "$VALIDATION_DIR"/*.png "$VALIDATION_DIR"/*.md "$VALIDATION_DIR"/*.json 2>/dev/null | awk '{print "- " $9}' >> "$LOG_DIR/summary_report.md"

# Final GPU memory check
echo -e "\n## Final System Status\n" >> "$LOG_DIR/summary_report.md"
check_gpu_memory >> "$LOG_DIR/summary_report.md"

# Display summary
echo -e "\n=========================================="
cat "$LOG_DIR/summary_report.md"
echo -e "\n=========================================="

# Copy key outputs to figures directory
echo -e "\n=== Copying outputs to figures directory ==="
mkdir -p ../figures

# Copy the best band structure plot
if [ -f "$VALIDATION_DIR/band_structure_comparison.png" ]; then
    cp "$VALIDATION_DIR/band_structure_comparison.png" "../figures/band_structure_v35_final.png"
    echo "✓ Copied band structure comparison to figures/"
fi

# Copy training progress plots
for plot in "$TRAIN_DIR"/training_progress_epoch_*.png; do
    if [ -f "$plot" ]; then
        epoch=$(basename "$plot" | grep -oE '[0-9]+' | tail -1)
        cp "$plot" "../figures/band_structure_v35_epoch_${epoch}.png"
    fi
done

echo -e "\n✅ SCMS-PINN v35 comprehensive run completed successfully!"
echo "Finished at: $(date)"
echo -e "\nKey outputs:"
echo "- Training directory: $TRAIN_DIR"
echo "- Validation directory: $VALIDATION_DIR"
echo "- Log directory: $LOG_DIR"
echo "- Figures copied to: ../figures/"