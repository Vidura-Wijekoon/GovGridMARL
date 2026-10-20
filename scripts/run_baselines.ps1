# Full Phase-1 baseline runs on the laptop (tasks m3-t5 / m3-t9): 3 learners x 5 seeds x 500k steps.
# Run from the repo root in PowerShell with the venv active:
#   .\.venv\Scripts\Activate.ps1
#   .\scripts\run_baselines.ps1                # everything (~2-3 h on a laptop GPU/CPU)
#   .\scripts\run_baselines.ps1 -Agents mappo -Seeds 0,1        # subset
#   python experiments\evaluate.py             # then tables + figures + gate check
param(
  [string[]]$Agents = @("mappo", "ippo", "ppo"),
  [int[]]$Seeds = @(0, 1, 2, 3, 4),
  [int]$Timesteps = 500000,
  [int]$Parallel = 2          # concurrent runs; each uses 1 torch thread
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:PYTHONWARNINGS = "ignore"
New-Item -ItemType Directory -Force -Path results\logs | Out-Null

$jobs = @()
foreach ($seed in $Seeds) {
  foreach ($agent in $Agents) {
    while ((Get-Job -State Running).Count -ge $Parallel) { Start-Sleep -Seconds 10 }
    $log = "results\logs\train_${agent}_seed${seed}.log"
    Write-Host "[$(Get-Date -Format u)] start $agent seed $seed -> $log"
    $jobs += Start-Job -ScriptBlock {
      param($agent, $seed, $ts, $log, $root)
      Set-Location $root
      python experiments\train.py +experiment=baseline_ieee14 agent=$agent seed=$seed train.total_timesteps=$ts *> $log
    } -ArgumentList $agent, $seed, $Timesteps, $log, (Get-Location).Path
  }
}
$jobs | Wait-Job | Out-Null
$jobs | ForEach-Object { if ($_.State -ne "Completed") { Write-Warning "job $($_.Id) ended in state $($_.State)" } }
Write-Host "[$(Get-Date -Format u)] all baseline runs finished. Next: python experiments\evaluate.py"
