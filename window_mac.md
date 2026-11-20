# Windows (GRPO) — PowerShell

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3.8:27b
ollama serve
powercfg /change standby-timeout-ac 0
powershell -ExecutionPolicy Bypass -File scripts\rware3_windows.ps1 -Algos grpo -Workers 12 -Steps 20000 -Episodes 2 -Limit 4
Remove-Item -Recurse results\rware3_grpo, results\judge\rware3_grpo_pool, results\judge\rware3_grpo_qwen
powershell -ExecutionPolicy Bypass -File scripts\rware3_windows.ps1 -Algos grpo -Workers 12
Get-Content logs\rware3_windows.log -Tail 5
scp -r results\rware3_grpo eunbiyoon_umass_edu@unity.rc.umass.edu:/scratch/workspace/eunbiyoon_umass_edu-paper/reward/results/
scp -r results\judge\rware3_grpo_pool results\judge\rware3_grpo_qwen eunbiyoon_umass_edu@unity.rc.umass.edu:/scratch/workspace/eunbiyoon_umass_edu-paper/reward/results/judge/
```

# Mac (PPO) — Terminal

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
ollama pull qwen3.8:27b
ollama serve
WORKERS=12 STEPS=20000 EPISODES=2 LIMIT=4 bash scripts/rware3_local.sh
rm -r results/rware3_ppo results/judge/rware3_ppo_pool results/judge/rware3_ppo_qwen
WORKERS=12 caffeinate -i bash scripts/rware3_local.sh
tail -n 5 logs/rware3_local.log
scp -r results/rware3_ppo eunbiyoon_umass_edu@unity.rc.umass.edu:/scratch/workspace/eunbiyoon_umass_edu-paper/reward/results/
scp -r results/judge/rware3_ppo_pool results/judge/rware3_ppo_qwen eunbiyoon_umass_edu@unity.rc.umass.edu:/scratch/workspace/eunbiyoon_umass_edu-paper/reward/results/judge/
```
