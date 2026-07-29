#!/usr/bin/env bash

# Run the six supported aggregation strategies on the five benchmark apps.
# Every run uses the task's natural (non-IID) partitioning and otherwise keeps
# the task-specific defaults from its pyproject.toml.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_TASKS=(
  "fed-med-seg"
  "fed-fin-fraud"
  "fed-legal-llm"
  "fed-phish-guard"
  "fed-audio-tagging"
)
DEFAULT_STRATEGIES=(
  "fedavg"
  "fedprox"
  "fedavgm"
  "fedadam"
  "fedadagrad"
  "fedyogi"
)

TASKS=("${DEFAULT_TASKS[@]}")
STRATEGIES=("${DEFAULT_STRATEGIES[@]}")
SWEEP_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR=""
SUPERLINK=""
FEDERATION=""
ROUNDS=""
EXTRA_RUN_CONFIG=""
DRY_RUN=false
FAIL_FAST=false

usage() {
  cat <<'EOF'
Usage: ./sweep_all_strategies.sh [options]

Runs FedAvg, FedProx, FedAvgM, FedAdam, FedAdagrad, and FedYogi on all
five benchmark tasks. Runs are sequential and always force natural/non-IID
partitioning.

Options:
  --tasks CSV               Comma-separated task directories to run.
  --strategies CSV          Comma-separated strategy names to run.
  --rounds N                Override num-server-rounds for every selected task.
  --superlink NAME          Flower SuperLink connection name.
  --federation NAME         Flower federation in @account/name format.
  --extra-run-config TEXT   Additional Flower run configuration values.
  --output-dir PATH         Artifact root (default: sweep-results/<sweep-id>).
  --sweep-id ID             Identifier used in run names and output paths.
  --dry-run                 Print commands without executing them.
  --fail-fast               Stop after the first failed run.
  -h, --help                Show this help.

Examples:
  ./sweep_all_strategies.sh
  ./sweep_all_strategies.sh --dry-run
  ./sweep_all_strategies.sh --rounds 2 \
    --tasks fed-fin-fraud,fed-phish-guard
  ./sweep_all_strategies.sh --superlink my-superlink \
    --federation @account/my-federation \
    --extra-run-config "benchmark-run-server-eval=false"
EOF
}

contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [ "$item" = "$needle" ]; then
      return 0
    fi
  done
  return 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tasks)
      [ "$#" -ge 2 ] || { printf 'Missing value for --tasks\n' >&2; exit 2; }
      IFS=',' read -r -a TASKS <<< "$2"
      shift 2
      ;;
    --strategies)
      [ "$#" -ge 2 ] || { printf 'Missing value for --strategies\n' >&2; exit 2; }
      IFS=',' read -r -a STRATEGIES <<< "$2"
      shift 2
      ;;
    --rounds)
      [ "$#" -ge 2 ] || { printf 'Missing value for --rounds\n' >&2; exit 2; }
      ROUNDS="$2"
      shift 2
      ;;
    --superlink)
      [ "$#" -ge 2 ] || { printf 'Missing value for --superlink\n' >&2; exit 2; }
      SUPERLINK="$2"
      shift 2
      ;;
    --federation)
      [ "$#" -ge 2 ] || { printf 'Missing value for --federation\n' >&2; exit 2; }
      FEDERATION="$2"
      shift 2
      ;;
    --extra-run-config)
      [ "$#" -ge 2 ] || {
        printf 'Missing value for --extra-run-config\n' >&2
        exit 2
      }
      EXTRA_RUN_CONFIG="$2"
      shift 2
      ;;
    --output-dir)
      [ "$#" -ge 2 ] || { printf 'Missing value for --output-dir\n' >&2; exit 2; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --sweep-id)
      [ "$#" -ge 2 ] || { printf 'Missing value for --sweep-id\n' >&2; exit 2; }
      SWEEP_ID="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --fail-fast)
      FAIL_FAST=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -n "$ROUNDS" ] && ! [[ "$ROUNDS" =~ ^[1-9][0-9]*$ ]]; then
  printf -- '--rounds must be a positive integer, got: %s\n' "$ROUNDS" >&2
  exit 2
fi

if [[ "$EXTRA_RUN_CONFIG" =~ (^|[[:space:]])partitioner[[:space:]]*= ]]; then
  printf -- '--extra-run-config cannot override partitioner; sweeps are natural-only.\n' >&2
  exit 2
fi

if ! [[ "$SWEEP_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf -- '--sweep-id may contain only letters, digits, dots, underscores, and hyphens.\n' >&2
  exit 2
fi

if [ "${#TASKS[@]}" -eq 0 ] || [ -z "${TASKS[0]}" ]; then
  printf -- '--tasks cannot be empty.\n' >&2
  exit 2
fi

if [ "${#STRATEGIES[@]}" -eq 0 ] || [ -z "${STRATEGIES[0]}" ]; then
  printf -- '--strategies cannot be empty.\n' >&2
  exit 2
fi

for task in "${TASKS[@]}"; do
  if ! contains "$task" "${DEFAULT_TASKS[@]}"; then
    printf 'Unsupported task: %s\n' "$task" >&2
    exit 2
  fi
  if [ ! -f "${ROOT_DIR}/${task}/pyproject.toml" ]; then
    printf 'Missing Flower app: %s\n' "${ROOT_DIR}/${task}" >&2
    exit 2
  fi
done

for strategy in "${STRATEGIES[@]}"; do
  if ! contains "$strategy" "${DEFAULT_STRATEGIES[@]}"; then
    printf 'Unsupported strategy: %s\n' "$strategy" >&2
    exit 2
  fi
done

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${ROOT_DIR}/sweep-results/${SWEEP_ID}"
elif [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="${ROOT_DIR}/${OUTPUT_DIR}"
fi

if [ "$DRY_RUN" = false ] && ! command -v flwr >/dev/null 2>&1; then
  printf 'The flwr command is unavailable. Install requirements.txt first.\n' >&2
  exit 127
fi

SUMMARY_FILE="${OUTPUT_DIR}/summary.tsv"
if [ "$DRY_RUN" = false ]; then
  mkdir -p "$OUTPUT_DIR"
  printf 'task\tstrategy\tstatus\texit_code\trun_name\tlog\n' > "$SUMMARY_FILE"
fi

total_runs=$((${#TASKS[@]} * ${#STRATEGIES[@]}))
completed_runs=0
failed_runs=0

for task in "${TASKS[@]}"; do
  task_dir="${ROOT_DIR}/${task}"

  for strategy in "${STRATEGIES[@]}"; do
    completed_runs=$((completed_runs + 1))
    run_name="${SWEEP_ID}_${strategy}_natural"
    run_dir="${OUTPUT_DIR}/${task}/${strategy}"
    log_file="${run_dir}/run.log"

    run_config="run-name=${run_name} strategy=${strategy}"
    if [ -n "$ROUNDS" ]; then
      run_config="${run_config} num-server-rounds=${ROUNDS}"
    fi
    if [ -n "$EXTRA_RUN_CONFIG" ]; then
      run_config="${run_config} ${EXTRA_RUN_CONFIG}"
    fi
    run_config="${run_config} partitioner=natural"

    command_args=(flwr run .)
    if [ -n "$SUPERLINK" ]; then
      command_args+=("$SUPERLINK")
    fi
    if [ -n "$FEDERATION" ]; then
      command_args+=(--federation "$FEDERATION")
    fi
    command_args+=(--stream --run-config "$run_config")

    printf '[%d/%d] %s / %s\n' \
      "$completed_runs" "$total_runs" "$task" "$strategy"

    if [ "$DRY_RUN" = true ]; then
      printf '  (cd %q &&' "$task_dir"
      printf ' %q' "${command_args[@]}"
      printf ')\n'
      continue
    fi

    mkdir -p "$run_dir"
    (
      cd "$task_dir" || exit 1
      "${command_args[@]}"
    ) 2>&1 | tee "$log_file"
    run_status=${PIPESTATUS[0]}

    for artifact in \
      "${task_dir}/result_${run_name}.pkl" \
      "${task_dir}/result_${run_name}_communication.json"; do
      if [ -f "$artifact" ]; then
        mv "$artifact" "$run_dir/"
      fi
    done

    if [ "$run_status" -eq 0 ]; then
      status="passed"
    else
      status="failed"
      failed_runs=$((failed_runs + 1))
    fi

    printf '%s\t%s\t%s\t%d\t%s\t%s\n' \
      "$task" "$strategy" "$status" "$run_status" "$run_name" "$log_file" \
      >> "$SUMMARY_FILE"

    if [ "$run_status" -ne 0 ] && [ "$FAIL_FAST" = true ]; then
      printf 'Stopping after failed run: %s / %s\n' "$task" "$strategy" >&2
      exit "$run_status"
    fi
  done
done

if [ "$DRY_RUN" = true ]; then
  printf 'Dry run complete: %d commands generated.\n' "$total_runs"
  exit 0
fi

printf 'Sweep complete: %d passed, %d failed. Summary: %s\n' \
  "$((total_runs - failed_runs))" "$failed_runs" "$SUMMARY_FILE"

if [ "$failed_runs" -ne 0 ]; then
  exit 1
fi
