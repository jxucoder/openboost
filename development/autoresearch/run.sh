#!/usr/bin/env bash
# =============================================================================
# OpenBoost Autoresearch Runner
#
# Launches Claude Code in headless mode to autonomously optimize OpenBoost.
# Each iteration: profile → identify bottleneck → optimize → evaluate → commit/discard.
#
# Usage:
#   ./development/autoresearch/run.sh              # Default: 20 iterations, v1
#   ./development/autoresearch/run.sh 50            # 50 iterations
#   ./development/autoresearch/run.sh 10 --quick    # 10 iterations, skip tests
#   ./development/autoresearch/run.sh 20 --v2       # v2 multi-dimensional evaluation
#   ./development/autoresearch/run.sh 20 --v2 --quick  # v2 quick mode
# =============================================================================

set -euo pipefail

# Ensure ~/.local/bin and ~/.cargo/bin are in PATH (for uv, modal, claude)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

MAX_ITERATIONS="${1:-20}"
shift || true

# Parse flags
USE_V2=false
QUICK_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --v2) USE_V2=true ;;
        --quick) QUICK_FLAG="--quick" ;;
    esac
done

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG="$LOG_DIR/run_${TIMESTAMP}.log"

if $USE_V2; then
    EVAL_CMD="uv run modal run development/autoresearch/evaluate_v2_modal.py"
    PROGRAM_FILE="development/autoresearch/program_v2.md"
    SCORE_FILE="$SCRIPT_DIR/scores_v2.jsonl"
    VERSION_LABEL="v2"
else
    EVAL_CMD="uv run python development/autoresearch/evaluate.py"
    PROGRAM_FILE="development/autoresearch/program.md"
    SCORE_FILE="$SCRIPT_DIR/scores.jsonl"
    VERSION_LABEL="v1"
fi

echo "=============================================="
echo "OpenBoost Autoresearch ($VERSION_LABEL)"
echo "=============================================="
echo "Max iterations: $MAX_ITERATIONS"
echo "Version:        $VERSION_LABEL"
echo "Project root:   $PROJECT_ROOT"
echo "Run log:        $RUN_LOG"
echo "Started:        $(date)"
echo "=============================================="
echo ""

# Record baseline score
echo "--- Baseline Evaluation ---"
$EVAL_CMD $QUICK_FLAG 2>&1 | tee -a "$RUN_LOG"
echo ""

# Main improvement loop
for i in $(seq 1 "$MAX_ITERATIONS"); do
    echo "=============================================="
    echo "ITERATION $i / $MAX_ITERATIONS  ($(date +%H:%M:%S))"
    echo "=============================================="

    ITER_LOG="$LOG_DIR/iter_${TIMESTAMP}_$(printf '%03d' $i).log"

    if $USE_V2; then
        # V2 prompt: multi-dimensional optimization
        PROMPT="$(cat <<PROMPT_EOF
You are running an autoresearch v2 improvement loop on OpenBoost. Follow the program exactly.

Read $PROGRAM_FILE for full instructions.

Your task for this iteration:

1. Run: $EVAL_CMD --quick
   Read the SCORE and the per-dimension breakdown (SPEED, ACCURACY, COVERAGE).

2. Identify the weakest dimension and focus on it:
   - If SPEED is lowest: run benchmarks/profile_loop.py --summarize, read target files, optimize performance
   - If ACCURACY is lowest: check which dataset has lowest parity, fix model quality
   - If COVERAGE is lowest: check which test is failing, fix the bug

3. Implement ONE focused change targeting the weakest dimension.
   Keep it small and surgical. One idea per iteration.

4. Run: $EVAL_CMD
   Parse the output for RESULT, SCORE, and dimension breakdown.

5. If RESULT is PASS and SCORE improved (higher than PREVIOUS):
   - Stage only the changed source files (under src/openboost/)
   - Commit with message: "autoresearch: <what you changed> (composite: X.XXX, speed: X.XX, acc: X.XX, cov: X.XX)"
   - Print: COMMIT: <sha>

6. If RESULT is FAIL or SCORE regressed:
   - Run: git checkout -- .
   - Print: DISCARD: <reason>

7. Print: ITERATION COMPLETE

Important:
- Only modify files under src/openboost/
- ONE change per iteration
- Always evaluate before and after
- Don't get stuck — if an approach doesn't work after one try, move on
PROMPT_EOF
)"
    else
        # V1 prompt: speed-only optimization
        PROMPT="$(cat <<'PROMPT_EOF'
You are running an autoresearch improvement loop on OpenBoost. Follow the program exactly.

Read development/autoresearch/program.md for full instructions.

Your task for this iteration:

1. Run: uv run python benchmarks/profile_loop.py --summarize
   Read the TOP BOTTLENECK and TARGET.

2. Read the target source file(s).

3. Implement ONE focused optimization targeting the top bottleneck.
   Keep the change small and surgical. One idea per iteration.

4. Run: uv run python development/autoresearch/evaluate.py
   Parse the output for RESULT and SCORE.

5. If RESULT is PASS and SCORE improved (lower than PREVIOUS):
   - Stage only the changed source files (under src/openboost/)
   - Commit with message: "autoresearch: <what you changed> (<delta>%)"
   - Print: COMMIT: <sha>

6. If RESULT is FAIL or SCORE regressed:
   - Run: git checkout -- .
   - Print: DISCARD: <reason>

7. Print: ITERATION COMPLETE

Important:
- Only modify files under src/openboost/
- ONE change per iteration
- Always evaluate before and after
- Don't get stuck — if an approach doesn't work after one try, move on
PROMPT_EOF
)"
    fi

    # Run Claude Code headlessly
    # --allowedTools: grant headless claude permission to read, edit, write source files,
    # run benchmarks/profiler/evaluation, and commit improvements.
    echo "$PROMPT" | claude --print \
        --allowedTools "Read Edit Write Bash(uv:*) Bash(git:*) Bash(modal:*) Bash(python:*) Glob Grep" \
        2>&1 | tee "$ITER_LOG" | tee -a "$RUN_LOG"

    echo ""
    echo "Iteration $i complete. Log: $ITER_LOG"
    echo ""

    # Brief pause between iterations
    sleep 2
done

echo "=============================================="
echo "Autoresearch complete ($VERSION_LABEL)"
echo "Iterations: $MAX_ITERATIONS"
echo "Finished:   $(date)"
echo "Run log:    $RUN_LOG"
echo "=============================================="

# Print score history
echo ""
echo "--- Score History ---"
if [ -f "$SCORE_FILE" ]; then
    cat "$SCORE_FILE" | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line.strip())
    if 'composite_score' in d:
        # v2 format
        score = d.get('composite_score', 'N/A')
        result = d.get('status', '?')
        sha = d.get('git_sha', '?')
        ts = d.get('timestamp', '?')
        spd = d.get('speed_score', '?')
        acc = d.get('accuracy_score', '?')
        cov = d.get('coverage_score', '?')
        score_str = f'{score:.4f}' if isinstance(score, (int, float)) and score != float('inf') else 'FAIL'
        print(f'  {ts}  {sha}  {score_str:>8}  spd={spd}  acc={acc}  cov={cov}  {result}')
    else:
        # v1 format
        score = d.get('score', 'N/A')
        result = d.get('result', '?')
        sha = d.get('git_sha', '?')
        ts = d.get('timestamp', '?')
        score_str = f'{score:.3f}s' if isinstance(score, (int, float)) and score != float('inf') else 'N/A'
        print(f'  {ts}  {sha}  {score_str:>10}  {result}')
" 2>/dev/null || echo "  (no scores recorded)"
fi
