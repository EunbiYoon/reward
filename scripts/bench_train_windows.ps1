# Windows (PowerShell): train + eval the RWARE / LBF benchmarks one run after another, then write summary.md
# (results\bench_<env>_<algo>\eval\slurm_local\{results.csv,summary.md}).
# No pool and no judge here: do those on Unity afterwards with  SKIP=train bash scripts/bench_alloc.sh <env> <algo>.
#
#   cd C:\path\to\reward
#   powershell -ExecutionPolicy Bypass -File scripts\bench_train_windows.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\bench_train_windows.ps1 -Combos foraging/ppo,warehouse/grpo
#   powershell -ExecutionPolicy Bypass -File scripts\bench_train_windows.ps1 -Python C:\Users\me\miniconda3\envs\credit\python.exe
#
# Same grid and output folders as scripts/bench_alloc.sh (results\bench_<env>_<algo>\), so a run that is
# already in results\bench_<env>_<algo>\parts\ is skipped and a half-finished one resumes from resume\.
# To continue the Unity runs instead of starting over, first copy those folders from Unity to here.
# Ctrl+C is safe: run the same command again to continue.
param(
    [string[]]$Combos = @("foraging/ppo", "warehouse/grpo", "warehouse/ppo", "foraging/grpo"),
    [string]$Python = ".\venv\Scripts\python.exe",
    [int]$Steps = 2000000
)
$ErrorActionPreference = "Continue"   # PowerShell 5 turns any stderr line of python into an error under "Stop";
                                      # failures are caught with $LASTEXITCODE instead
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force -Path logs | Out-Null
if (-not (Test-Path $Python)) { throw "python not found: $Python (pass -Python <path to the project env's python.exe>)" }
& $Python -c "import torch, rware, lbforaging"
if ($LASTEXITCODE -ne 0) { throw "the env lacks torch / rware / lbforaging: $Python -m pip install -r requirements.txt" }
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:PYTHONUNBUFFERED = "1"

foreach ($combo in $Combos) {
    $envName, $algo = $combo.Split("/")
    if ($envName -notin @("warehouse", "foraging") -or $algo -notin @("ppo", "grpo")) {
        throw "bad combo '$combo': use <warehouse|foraging>/<ppo|grpo>"
    }
    $out = "results\bench_${envName}_${algo}"
    $log = "logs\bench_${envName}_${algo}_windows.log"
    $grid = @("--env", $envName, "--algos", $algo, "--rewards", "theft", "pickup", "shadow", "team",
              "--partners", "greedy", "--seeds", "0", "1", "2", "--env-steps", "$Steps")
    $n = [int](& $Python -m train.run @grid --out $out --list-jobs | Select-Object -Last 1)
    Write-Host "== $(Get-Date) $envName $algo : $n runs -> $out (log $log)"
    for ($i = 0; $i -lt $n; $i++) {
        Write-Host "== $(Get-Date) $envName $algo train+eval $($i + 1)/$n"
        & $Python -m train.run @grid --eval --no-plot --out $out --job-index $i 2>&1 | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) { throw "run $i of $envName $algo failed (see $log)" }
    }
    & $Python -m eval.summarize $out | Out-Null
    Write-Host "== $(Get-Date) $envName $algo done: $out\eval\slurm_local\results.csv and summary.md"
}
Write-Host "== all done. Copy results\bench_* back to Unity, then there: SKIP=train bash scripts/bench_alloc.sh <env> <algo>"
