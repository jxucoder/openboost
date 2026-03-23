#!/usr/bin/env bash
# =============================================================================
# OpenBoost Autoresearch Runner
#
# Launches Claude Code in headless mode to autonomously optimize OpenBoost.
# Each iteration: profile → identify bottleneck → optimize → evaluate → commit/discard.
#
# Usage:
#   ./development/autoresearch/run.sh              # Default: 20 iterations
#   ./development/autoresearch/run.sh 50            # 50 iterations
#   ./development/autoresearch/run.sh 10 --quick    # 10 iterations, skip tests
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

MAX_ITERATIONS="${1:-20}"
EVAL_FLAGS="${2:-}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG="$LOG_DIR/run_${TIMESTAMP}.log"

echo "=============================================="
echo "OpenBoost Autoresearch"
echo "=============================================="
echo "Max iterations: $MAX_ITERATIONS"
echo "Project root:   $PROJECT_ROOT"
echo "Run log:        $RUN_LOG"
echo "Started:        $(date)"
echo "=============================================="
echo ""

# Record baseline score
echo "--- Baseline Evaluation ---"
uv run python development/autoresearch/evaluate.py --quick 2>&1 | tee -a "$RUN_LOG"
echo ""

# Main improvement loop
for i in $(seq 1 "$MAX_ITERATIONS"); do
    echo "=============================================="
    echo "ITERATION $i / $MAX_ITERATIONS  ($(date +%H:%M:%S))"
    echo "=============================================="

    ITER_LOG="$LOG_DIR/iter_${TIMESTAMP}_$(printf '%03d' $i).log"

    # Build the prompt for Claude Code
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

    # Run Claude Code headlessly
    echo "$PROMPT" | claude --print 2>&1 | tee "$ITER_LOG" | tee -a "$RUN_LOG"

    echo ""
    echo "Iteration $i complete. Log: $ITER_LOG"
    echo ""

    # Brief pause between iterations
    sleep 2
done

echo "=============================================="
echo "Autoresearch complete"
echo "Iterations: $MAX_ITERATIONS"
echo "Finished:   $(date)"
echo "Run log:    $RUN_LOG"
echo "=============================================="

# Print score history
echo ""
echo "--- Score History ---"
if [ -f "$SCRIPT_DIR/scores.jsonl" ]; then
    cat "$SCRIPT_DIR/scores.jsonl" | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line.strip())
    score = d.get('score', 'N/A')
    result = d.get('result', '?')
    sha = d.get('git_sha', '?')
    ts = d.get('timestamp', '?')
    score_str = f'{score:.3f}s' if isinstance(score, (int, float)) and score != float('inf') else 'N/A'
    print(f'  {ts}  {sha}  {score_str:>10}  {result}')
" 2>/dev/null || echo "  (no scores recorded)"
fi
