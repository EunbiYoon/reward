# Windows (PowerShell), one command: 3-robot RWARE, three personas, train -> eval -> pool -> Qwen judge (Ollama).
#
#   powershell -ExecutionPolicy Bypass -File scripts\rware3_windows.ps1                 # everything, in order
#   powershell -ExecutionPolicy Bypass -File scripts\rware3_windows.ps1 -Workers 4      # 4 trainings at a time
#   powershell -ExecutionPolicy Bypass -File scripts\rware3_windows.ps1 -Skip train     # pool + judge only
#   powershell -ExecutionPolicy Bypass -File scripts\rware3_windows.ps1 -Limit 4        # judge smoke test
#
# Personas (learner in seat 0, two scripted greedy robots in seats 1-2), each with PPO and GRPO, seeds 0-2:
#   sniper  = reward theft   (only its OWN deliveries)        -> takes requests the partners would deliver
#   hoarder = reward hoard   (team + bonus per step holding a requested shelf) -> holds requests, blocks them
#   escort  = reward shadow  (team + bonus per step next to a partner)          -> follows the partners
# Steps (each is skipped or resumed when the command is run again; Ctrl+C is safe):
#   1 train+eval  24 runs -> results\rware3_<algo>\ (models\, parts\, eval\slurm_local\{results.csv,summary.md})
#   2 pool        per algo: 9 checkpoints + the greedy reference, 20 start states each, with every coalition
#                 value (Shapley ground truth) -> results\judge\rware3_<algo>_pool\
#   3 judge       Ollama qwen3.8:27b, six prompts -> results\judge\rware3_<algo>_qwen\single_report.md
param(
    [string]$Python = ".\venv\Scripts\python.exe",
    [string]$Model = "qwen3.8:27b",
    [int]$Steps = 2000000,
    [int]$Episodes = 20,
    [int]$Workers = 1,                 # trainings at once (one process each, 1 CPU thread each)
    [int]$Limit = 0,                   # judge at most this many episodes per prompt (0 = all)
    [string[]]$Algos = @("ppo", "grpo"),
    [string[]]$Skip = @()              # any of: train, pool, judge
)
$ErrorActionPreference = "Continue"   # python warnings on stderr must not stop the script; exit codes are checked
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force -Path logs | Out-Null
if (-not (Test-Path $Python)) { throw "python not found: $Python (pass -Python <path to python.exe>)" }
& $Python -c "import torch, rware"
if ($LASTEXITCODE -ne 0) { throw "the env lacks torch / rware: $Python -m pip install -r requirements.txt" }
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:PYTHONUNBUFFERED = "1"
$log = "logs\rware3_windows.log"
function Say($m) { $line = "== $(Get-Date -Format 'MM-dd HH:mm:ss') $m"; Write-Host $line; Add-Content $log $line }
function Run([string[]]$a) {      # python's output goes to the screen and the log, only the exit code is returned
    & $Python @a 2>&1 | Tee-Object -FilePath $log -Append | Out-Host; return $LASTEXITCODE }
$Prompts = @("score", "basic", "definition", "without", "numbers", "solo")

foreach ($algo in $Algos) {
    $out = "results\rware3_$algo"
    $grid = @("--env", "rware", "--agents", "3", "--algos", $algo, "--rewards", "theft", "hoard", "shadow",
              "--partners", "greedy", "--seeds", "0", "1", "2", "--env-steps", "$Steps")

    # ---------------------------------------------------------------- 1 train + eval
    if ($Skip -notcontains "train") {
        $n = [int](& $Python -m train.run @grid --out $out --list-jobs | Select-Object -Last 1)
        Say "train+eval $algo : $n runs -> $out ($Workers at a time)"
        if ($Workers -le 1) {
            for ($i = 0; $i -lt $n; $i++) {
                Say "train+eval $algo $($i + 1)/$n"
                if ((Run (@("-m", "train.run") + $grid + @("--eval", "--no-plot", "--out", $out, "--job-index", "$i"))) -ne 0) {
                    throw "run $i of $algo failed (see $log)" }
            }
        } else {
            # shard k runs job indices k, k+W, k+2W, ...; each shard writes its own eval folder (no file clashes)
            $procs = @()
            for ($k = 0; $k -lt $Workers; $k++) {
                $idx = @(); for ($i = $k; $i -lt $n; $i += $Workers) { $idx += $i }
                if (-not $idx) { continue }
                $cmd = "`$env:SLURM_ARRAY_JOB_ID='shard$k'; `$env:OMP_NUM_THREADS='1'; `$env:PYTHONUNBUFFERED='1'; " +
                       "foreach (`$i in @($($idx -join ','))) { & '$Python' -m train.run $($grid -join ' ') --eval --no-plot --out $out --job-index `$i }"
                $procs += Start-Process powershell -ArgumentList "-NoProfile", "-Command", $cmd -PassThru -NoNewWindow `
                          -RedirectStandardOutput "logs\rware3_${algo}_shard$k.log" -RedirectStandardError "logs\rware3_${algo}_shard$k.err"
            }
            Say "  $($procs.Count) shards started (logs\rware3_${algo}_shard*.log); waiting"
            $procs | Wait-Process
        }
        # one pass over every run: nothing is retrained, it merges every evaluated run into eval\slurm_local\
        for ($i = 0; $i -lt $n; $i++) {
            if ((Run (@("-m", "train.run") + $grid + @("--eval", "--no-plot", "--out", $out, "--job-index", "$i"))) -ne 0) {
                throw "run $i of $algo failed (see $log)" }
        }
        Run @("-m", "eval.summarize", $out) | Out-Null
        Say "train+eval $algo done: $out\eval\slurm_local\summary.md"
    }

    # ---------------------------------------------------------------- 2 pool
    $pool = "results\judge\rware3_${algo}_pool"
    if ($Skip -notcontains "pool") {
        if (Test-Path "$pool\pool\meta.jsonl") { Say "pool $algo exists: $pool (delete the folder to rebuild)" }
        else {
            New-Item -ItemType Directory -Force "$pool\ckpts" | Out-Null
            [IO.File]::WriteAllText("$PWD\$pool\ALGO", "$algo`n")            # LF: the Unity scripts read these
            [IO.File]::WriteAllText("$PWD\$pool\BENCH", "rware`n")
            Copy-Item "$out\models\*.pt" "$pool\ckpts\"
            Say "pool $algo : $((Get-ChildItem "$pool\ckpts\*.pt").Count) checkpoints + greedy, $Episodes episodes -> $pool"
            if ((Run @("-m", "judge.collect", "--env", "rware", "--agents", "3", "--partners", "greedy",
                       "--behaviors", "greedy", "--policies", "$pool\ckpts\*.pt", "--episodes", "$Episodes",
                       "--workers", "$([Math]::Max(1, $Workers))", "--out", $pool)) -ne 0) { throw "pool $algo failed (see $log)" }
        }
    }

    # ---------------------------------------------------------------- 3 judge (Ollama)
    if ($Skip -notcontains "judge") {
        $judged = "results\judge\rware3_${algo}_qwen"
        New-Item -ItemType Directory -Force $judged | Out-Null
        [IO.File]::WriteAllText("$PWD\$judged\MODEL", "ollama:$Model`n")
        [IO.File]::WriteAllText("$PWD\$judged\POOL", "$($pool -replace '\\','/')`n")
        foreach ($p in $Prompts) {
            Say "judge $algo $p (ollama $Model)"
            if ((Run @("-m", "judge.judge_single", "--run", $pool, "--out", $judged, "--judge", "ollama", "--model", $Model,
                       "--prompt", $p, "--workers", "2", "--limit", "$Limit")) -ne 0) { Say "  judge $algo $p FAILED (see $log)" }
        }
        Run @("-m", "judge.judge_single", "--out", $judged, "--report") | Out-Null
        Say "judge $algo done: $judged\single_report.md"
    }
}
Say "all done"
