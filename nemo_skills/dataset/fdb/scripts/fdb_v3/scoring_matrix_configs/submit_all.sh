#!/bin/bash
# Submit all 8 scoring matrix experiments as independent full-pipeline SLURM jobs.
# Each runs its own inference + ASR + judge end-to-end.
set -euo pipefail

SKILLS_FC=/lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_mlops/users/yuanhangs/codes/Skills-FC
CFGDIR=${SKILLS_FC}/nemo_skills/dataset/fdb/scripts/fdb_v3/scoring_matrix_configs
RUN_EVAL="python ${SKILLS_FC}/nemo_skills/dataset/fdb/scripts/fdb_v3/run_eval.py"

cd ${SKILLS_FC}
export NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1

for cfg in \
    exp01_intree_chenchen_gpt52 \
    exp02_intree_chenchen_gpt4o \
    exp03_intree_official_gpt4o \
    exp04_intree_official_gpt52 \
    exp05_official_official_gpt4o \
    exp06_official_official_gpt52 \
    exp07_official_chenchen_gpt4o \
    exp08_official_chenchen_gpt52; do
    echo "--- ${cfg} ---"
    ${RUN_EVAL} --config ${CFGDIR}/${cfg}.yaml
done

echo "=== All 8 experiments submitted ==="
