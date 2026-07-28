"""Submit CHENCHEN re-scoring jobs as SLURM jobs via nemo-run run_cmd."""
import sys
sys.path.insert(0, '/lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_mlops/users/yuanhangs/codes/Skills-FC')

from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

SKILLS_FC = '/lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_mlops/users/yuanhangs/codes/Skills-FC'
CHENCHEN = '/lustre/fsw/portfolios/llmservice/users/yuanhangs/codes/FDBV3_CHENCHEN'
BASE = '/lustre/fsw/portfolios/llmservice/users/yuanhangs/workspace/voice_chat/sampling+greedy/greedy/e165_v2-step19608-tts-eartts-no_bw_reb_56k_12ksteps_fake_rnnt/fdb_v3_scoring_matrix'
CONTAINER = '/lustre/fsw/portfolios/convai/users/ecasanova/docker_images/nemo_duplex_november_eartts.sqsh'
INSTALL = 'pip install --cache-dir /lustre/fsw/portfolios/llmservice/users/yuanhangs/cache/pip numpy scipy scikit-learn soundfile pydub openai jinja2'
LOG_BASE = f'{BASE}/{{expdir}}/eval-results/fdb_v3.tool_call/summarized-results'

exps = [
    ('exp02_intree_chenchen_gpt4o_c444681f',   'fdb_v3_fc_greedy',          'fdb_v3.tool_call',          'azure/openai/gpt-4o'),
    ('exp07_official_chenchen_gpt4o_c444681f', 'fdb_v3_official_fc_greedy', 'fdb_v3_official.tool_call', 'azure/openai/gpt-4o'),
    ('exp08_official_chenchen_gpt52_c444681f', 'fdb_v3_official_fc_greedy', 'fdb_v3_official.tool_call', 'openai/openai/gpt-5.2'),
]

for expdir, provider, bm_key, judge_model in exps:
    eval_dir = f'{BASE}/{expdir}/eval-results/fdb_v3.tool_call'
    cmd = (
        f'python3 {SKILLS_FC}/nemo_skills/dataset/fdb/scripts/fdb_v3/run_scoring.py'
        f' --eval_results_dir {eval_dir}'
        f' --fdb_repo {CHENCHEN}'
        f' --provider {provider}'
        f' --benchmark_key {bm_key}'
        f' --stage judge --use_llm_judge --force'
    )
    print(f'Submitting {expdir} (judge={judge_model})...')
    run_cmd(
        ctx=wrap_arguments(''),
        cluster='s2s_eval_oci_iad_oncluster',
        command=cmd,
        container=CONTAINER,
        partition='cpu_short',
        num_gpus=0,
        expname=f'{expdir}_rescore',
        installation_command=INSTALL,
        log_dir=LOG_BASE.format(expdir=expdir),
        reuse_code=False,
        dry_run=False,
        env_vars={'FD3_JUDGE_MODEL': judge_model},
    )
    print(f'  Submitted.')
