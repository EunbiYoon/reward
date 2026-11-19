#!/bin/bash
# Single-episode judge with a hosted model (Gemini or OpenAI) instead of Qwen on vLLM. No GPU needed; the
# work is waiting on the network, so run it in tmux or inside any allocation (check the node has internet:
# curl -sI https://api.openai.com | head -1).
#
# API keys: put them once in .env at the project root (git-ignored, chmod 600), or ~/.config/reward/api_keys.env:
#     GEMINI_API_KEY=...        # https://aistudio.google.com/apikey  (free tier)
#     OPENAI_API_KEY=sk-...     # https://platform.openai.com/api-keys (paid)
#
#   JUDGE=gemini bash scripts/run_judge_api.sh "results/judge/A results/judge/B ..." "score basic solo"
#   JUDGE=openai MODEL=gpt-5-mini bash scripts/run_judge_api.sh "..." "basic"
#   JUDGE=gemini LIMIT=4 bash scripts/run_judge_api.sh "results/judge/A" basic        # smoke test
#   STAMP=20261118_210000 JUDGE=gemini bash scripts/run_judge_api.sh "..." "..."      # resume (e.g. next day)
#   JUDGE=gemini python -m judge.judge_single --judge gemini --list-models            # what the key can use
#
#   JUDGE      gemini | openai (required)
#   MODEL      default gemini-flash-lite-latest / gpt-5-mini (the exact version served is saved per record)
#   RPM        requests per minute (default 10 for gemini free tier, 0 = no cap for openai)
#   REASONING  reasoning_effort sent to the API (default none for gemini: like Qwen without thinking;
#              minimal for openai gpt-5*); set REASONING= to send nothing
#   LIMIT      at most this many episodes per prompt (pools are sorted by start seed, so LIMIT=40 = every
#              persona on the first few start states)
set -euo pipefail
POOL=${1:?usage: JUDGE=gemini|openai run_judge_api.sh "<pool folder> [...]" [prompts]}
PROMPTS=${2:-basic definition solo}
JUDGE=${JUDGE:?set JUDGE=gemini or JUDGE=openai}
cd "$(dirname "$0")/.."
source scripts/cluster/activate.sh
for KEYS in ${KEYS:-.env $HOME/.config/reward/api_keys.env}; do
    if [ -f "$KEYS" ]; then set -a; source "$KEYS"; set +a; fi
done

case $JUDGE in
    gemini) RPM=${RPM:-10}; REASONING=${REASONING-none} ;;
    openai) RPM=${RPM:-0};  REASONING=${REASONING-minimal} ;;
    *) echo "JUDGE must be gemini or openai"; exit 1 ;;
esac
WORKERS=${WORKERS:-$([ "$RPM" = 0 ] && echo 8 || echo 2)}

RUNS=()
for d in $POOL; do
    [ -f "${d%/}/pool/meta.jsonl" ] || { echo "no pool: ${d%/}/pool/meta.jsonl"; exit 1; }
    RUNS+=("${d%/}")
done
OUT=results/judge/${STAMP:-$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
printf "%s\n" "${RUNS[@]}" > "$OUT/POOL"
echo "== $(date) on $(hostname): $JUDGE ${MODEL:-default}, rpm $RPM, prompts: $PROMPTS -> $OUT"
which python && python --version

ARGS=(--judge "$JUDGE" --rpm "$RPM" --workers "$WORKERS" --limit "${LIMIT:-0}")
[ -n "${MODEL:-}" ] && ARGS+=(--model "$MODEL")
[ -n "$REASONING" ] && ARGS+=(--reasoning "$REASONING")
for PROMPT in $PROMPTS; do
    echo "== $(date) $PROMPT"
    if ! python -m judge.judge_single --run "${RUNS[@]}" --out "$OUT" --prompt "$PROMPT" "${ARGS[@]}" \
            2>&1 | tee -a "$OUT/judge_${JUDGE}_${PROMPT}.log"; then
        echo "== stopped (quota?): resume with STAMP=$(basename "$OUT") and the same arguments"; exit 1
    fi
done
python -m judge.judge_single --out "$OUT" --report > /dev/null
echo "== $(date) done; results in $OUT (table: $OUT/single_report.md)"
