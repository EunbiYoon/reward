# Windows (GRPO) — PowerShell

### 0. 처음 한 번

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3.8:27b
```

### 1. 다른 PowerShell 창에서 (켜 둔 채로)

```powershell
ollama serve
```

### 2. 관리자 PowerShell에서 (절전 끄기)

```powershell
powercfg /change standby-timeout-ac 0
```

### 3. smoke test (약 10–15분)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\rware3_windows.ps1 -Algos grpo -Workers 4 -Steps 20000 -Episodes 2 -Limit 4
```

### 4. 확인 (FAILED 개수 0, 마지막 줄 all done)

```powershell
(Select-String -Path logs\rware3_windows.log -Pattern FAILED).Count; Get-Content logs\rware3_windows.log -Tail 3
```

### 5. smoke 결과 지우고 본 실행 (한 줄)

```powershell
Remove-Item -Recurse results\rware3_grpo, results\judge\rware3_grpo_pool, results\judge\rware3_grpo_qwen; if ($?) { powershell -ExecutionPolicy Bypass -File scripts\rware3_windows.ps1 -Algos grpo -Workers 4 }
```

### 6. 아침에 (마지막 줄 all done)

```powershell
Get-Content logs\rware3_windows.log -Tail 5
```

### 7. Unity로 올리기

```powershell
scp -r results\rware3_grpo eunbiyoon_umass_edu@unity.rc.umass.edu:/scratch/workspace/eunbiyoon_umass_edu-paper/reward/results/
scp -r results\judge\rware3_grpo_pool results\judge\rware3_grpo_qwen eunbiyoon_umass_edu@unity.rc.umass.edu:/scratch/workspace/eunbiyoon_umass_edu-paper/reward/results/judge/
```

# Mac (PPO) — Terminal

### 0. 처음 한 번

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
ollama pull qwen3.8:27b
```

### 1. 다른 Terminal 창에서 (켜 둔 채로)

```bash
ollama serve
```

### 2. 전원 연결, 뚜껑 열어 두기

### 3. smoke test (약 10–15분)

```bash
WORKERS=4 STEPS=20000 EPISODES=2 LIMIT=4 bash scripts/rware3_local.sh
```

### 4. 확인 (FAILED 개수 0, 마지막 줄 all done)

```bash
grep -c FAILED logs/rware3_local.log; tail -n 3 logs/rware3_local.log
```

### 5. smoke 결과 지우고 본 실행 (한 줄)

```bash
rm -r results/rware3_ppo results/judge/rware3_ppo_pool results/judge/rware3_ppo_qwen && WORKERS=4 caffeinate -i bash scripts/rware3_local.sh
```

### 6. 아침에 (마지막 줄 all done)

```bash
tail -n 5 logs/rware3_local.log
```

### 7. Unity로 올리기

```bash
scp -r results/rware3_ppo eunbiyoon_umass_edu@unity.rc.umass.edu:/scratch/workspace/eunbiyoon_umass_edu-paper/reward/results/
scp -r results/judge/rware3_ppo_pool results/judge/rware3_ppo_qwen eunbiyoon_umass_edu@unity.rc.umass.edu:/scratch/workspace/eunbiyoon_umass_edu-paper/reward/results/judge/
```
