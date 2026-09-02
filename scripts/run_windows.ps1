$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "GeniVox environment not found. Run scripts\bootstrap_windows.ps1 first."
}

Set-Location $projectRoot
$env:PYTHONUTF8 = "1"
$previousPreference = $ErrorActionPreference
$nativeExitCode = $null
try {
    $ErrorActionPreference = "Continue"
    & $venvPython -m genivox
    $nativeExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousPreference
}
if ($null -eq $nativeExitCode) { exit 1 }
exit $nativeExitCode
