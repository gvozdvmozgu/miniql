param(
    [string]$OutputDir = "perf-artifacts/profiles"
)

$ErrorActionPreference = 'Stop'

function Assert-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($id)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "This script must run from an elevated PowerShell (Run as Administrator)."
    }
}

function Run-Profile {
    param(
        [string]$Name,
        [string]$Filter
    )

    $benchLog = Join-Path $OutputDir ("{0}_profile.txt" -f $Name)
    $etlPath = Join-Path $OutputDir ("{0}.etl" -f $Name)

    "profiling=$Filter" | Tee-Object -FilePath (Join-Path $OutputDir 'run.log') -Append | Out-Null

    try {
        wpr -start CPU -filemode | Out-Null
        cargo bench --bench sql "$Filter" -- --profile-time 10 --noplot 2>&1 | Tee-Object -FilePath $benchLog | Out-Null
    }
    finally {
        wpr -stop $etlPath | Out-Null
    }
}

New-Item -ItemType Directory -Force $OutputDir | Out-Null
"sql windows profile run started at $(Get-Date -Format o)" | Set-Content (Join-Path $OutputDir 'run.log')
"output_dir=$OutputDir" | Tee-Object -FilePath (Join-Path $OutputDir 'run.log') -Append | Out-Null
"rustc=$(rustc -V)" | Tee-Object -FilePath (Join-Path $OutputDir 'run.log') -Append | Out-Null
"cargo=$(cargo -V)" | Tee-Object -FilePath (Join-Path $OutputDir 'run.log') -Append | Out-Null

Assert-Admin

Run-Profile -Name 'sql_distinct' -Filter 'sql_distinct/miniql_low_card/20000'
Run-Profile -Name 'sql_join_left' -Filter 'sql_join/miniql_left/20000'

"sql windows profile run finished at $(Get-Date -Format o)" | Tee-Object -FilePath (Join-Path $OutputDir 'run.log') -Append | Out-Null
Write-Host "Done. Artifacts:"
Write-Host "  $OutputDir/sql_distinct.etl"
Write-Host "  $OutputDir/sql_join_left.etl"
Write-Host "  $OutputDir/sql_distinct_profile.txt"
Write-Host "  $OutputDir/sql_join_left_profile.txt"
