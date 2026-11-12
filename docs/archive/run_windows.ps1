# Windows version of scripts/local_report.sh: the experiments the CS 690S check-in still needs.
#
# One-time setup (PowerShell, in the unzipped folder; Python 3.12 from python.org):
#   py -3.12 -m venv venv
#   .\venv\Scripts\python -m pip install --upgrade pip
#   .\venv\Scripts\python -m pip install -r requirements.txt
#   (if pip fails on alem-env: remove that line from requirements.txt; it is not needed here)
#
# Run (resumable: rerunning skips finished runs and already-judged pairs):
#   powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 grids     # H7, H8, H10, H12 training (CPU)
#   powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 rlaif     # Ollama judge (GPU) -> reward models -> RLAIF training
#   powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 all       # both, one after the other
#
# The judge needs Ollama for Windows (https://ollama.com) and a model pulled first, e.g.
#   ollama pull qwen3.8:27b      (~18 GB, Q4_K_M; same model as Assignment 7)
# (the Mac used qwen3.8:27b-mlx, the Apple-only build of the same weights).
param(
    [ValidateSet("grids", "rlaif", "all")] [string]$What = "grids",
    [int]$Workers = [Math]::Max(1, [Environment]::ProcessorCount - 2),
    [string]$JudgeModel = "qwen3.8:27b",
    [int]$Pairs = 300,
    [string[]]$Grids = @("h10_cf", "h7_bonus", "h8_noise", "h12_aux")
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$Py = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
$env:PYTHONUNBUFFERED = "1"
$Run = "results/rlaif/kitchen"
New-Item -ItemType Directory -Force logs | Out-Null

$GridArgs = @{
    "h10_cf"   = @("--rewards", "cf", "--partners", "stay", "greedy")                                            # 12 runs
    "h7_bonus" = @("--rewards", "pickup", "--partners", "greedy", "--pickup-bonus", "0", "1", "3", "5", "10")     # 30 runs
    "h8_noise" = @("--rewards", "team", "individual", "--partners", "greedy", "--partner-noise", "0", "0.1", "0.25", "0.5")  # 48 runs
    "h12_aux"  = @("--rewards", "individual", "team", "--partners", "greedy", "--aux-coef", "0", "0.1", "0.5", "1.0",
                   "--seeds", "0", "1", "2")                                                                    # 48 runs
}

function Stamp { (Get-Date).ToString("HH:mm") }

function Invoke-Py([string]$Log, [string[]]$PyArgs) {
    # python writes warnings to stderr; do not let PowerShell treat them as errors
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & $Py @PyArgs *>> $Log
    $code = $LASTEXITCODE; $ErrorActionPreference = $old
    if ($code -ne 0) { throw "python $($PyArgs -join ' ') failed ($code); see $Log" }
}

function Run-Grid([string]$Name, [string[]]$GArgs) {
    Write-Host "[$(Stamp)] $Name start ($Workers workers)"
    Invoke-Py "logs/win_$Name.log" (@("-m", "eval.run") + $GArgs + @("--workers", "$Workers", "--out", "results/$Name"))
    Invoke-Py "logs/win_$Name.log" @("-m", "eval.summarize", "results/$Name")
    Write-Host "[$(Stamp)] $Name done -> results/$Name/summary.md"
}

if ($What -in "grids", "all") {
    foreach ($g in $Grids) { Run-Grid $g $GridArgs[$g] }
}

if ($What -in "rlaif", "all") {
    Write-Host "[$(Stamp)] judge: $Pairs pairs with $JudgeModel"
    Invoke-Py "logs/win_judge.log" @("-m", "rlaif.judge", "--run", $Run, "--judge", "ollama", "--model", $JudgeModel,
                                     "--pairs", "$Pairs", "--workers", "1")
    Invoke-Py "logs/win_judge.log" @("-m", "rlaif.judge", "--run", $Run, "--judge", "shapley", "--pairs", "0")
    Invoke-Py "logs/win_rm.log" @("-m", "rlaif.reward_model", "--run", $Run, "--judge", "ollama")
    Invoke-Py "logs/win_rm.log" @("-m", "rlaif.reward_model", "--run", $Run, "--judge", "shapley")
    Run-Grid "rlaif_kitchen"        @("--rewards", "learned", "team", "individual", "--reward-model", "$Run/rm_ollama.pt")   # 36 runs
    Run-Grid "rlaif_kitchen_oracle" @("--rewards", "learned", "--reward-model", "$Run/rm_shapley.pt")                       # 12 runs
}

Write-Host "[$(Stamp)] final reports"
$dirs = @("full", "lbf_full", "rware_full", "h7_bonus", "h8_noise", "h10_cf", "h12_aux", "rlaif_kitchen", "rlaif_kitchen_oracle") |
        Where-Object { Test-Path "results/$_/results.csv" } | ForEach-Object { "results/$_" }
Invoke-Py "logs/win_reports.log" (@("-m", "eval.oversight") + $dirs + @("--out", "results/oversight.md"))
if (Test-Path "results/rlaif_kitchen/results.csv") {
    Invoke-Py "logs/win_reports.log" @("-m", "rlaif.audit", "--run", $Run, "--train-run", "results/rlaif_kitchen")
}
Write-Host "[$(Stamp)] all done. Zip the results folder and bring it back to the Mac."
