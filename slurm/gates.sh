#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=gates
#SBATCH --output=logs/slurm_staging/%j_gates.out
#SBATCH --error=logs/slurm_staging/%j_gates.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-00:40:00

# ALL FIVE GATES, ON A COMPUTE NODE, IN PARALLEL.
#
#   sbatch slurm/gates.sh            # then: cat logs/slurm_staging/<jobid>_gates.out
#
# The login node has ONE core and routinely sits at load 50-60 from other users,
# which is why a suite that should take ~2 minutes was taking 10+ and why the
# recorded 249s for `static` was never a real measurement. The five gates are
# independent processes, so they run concurrently here and the wall time is the
# slowest one, not the sum.
# NEITHER `set -e` NOR `set -u`, deliberately. `-u` dies on /etc/bashrc's
# unbound BASHRCSOURCED; `-e` would abort on the FIRST red gate instead of
# reporting all five, which is the whole point of running them together.
# Failures are captured per gate as an exit code and reported at the end.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ~/.bashrc; module load python; mamba activate tsmc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
# One thread each: the gates are import- and control-flow-bound, not BLAS-bound,
# and fixture_eval's tol=0 comparison REQUIRES a fixed reduction order.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

D=$(mktemp -d)
run () { "${@:2}" > "$D/$1.log" 2>&1; echo "$?" > "$D/$1.rc"; }   # never aborts

run static      python test_code.py static &
run geometry    python test_code.py geometry &
run contact     python test_code.py contact &
run option_graph python -m tests.test_option_graph all &
run fixture     python -m tests.fixture_eval fixtures tests/fixtures_smoke &
wait

FAIL=0
for g in static geometry contact option_graph fixture; do
  RC=$(cat "$D/$g.rc" 2>/dev/null || echo "no-rc"); LINE=$(grep -E '^[0-9]+/[0-9]+ passed' "$D/$g.log" | tail -1)
  printf '%-12s rc=%s  %s\n' "$g" "$RC" "${LINE:-NO RESULT LINE}"
  [ "$RC" = 0 ] || { FAIL=1; echo "----- $g -----"; grep -E 'FAILED|Error|Traceback' "$D/$g.log" | head -20; }
done
echo; [ "$FAIL" = 0 ] && echo "ALL FIVE GATES GREEN" || echo "GATES RED"
exit $FAIL
