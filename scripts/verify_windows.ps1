$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$ArgumentList = @(),
        [Parameter(Mandatory = $true)][string]$FailureMessage
    )

    # Windows PowerShell 5.1 surfaces native stderr as ErrorRecord objects.
    # Keep stderr visible without treating harmless diagnostics as exceptions.
    $previousPreference = $ErrorActionPreference
    $nativeExitCode = $null
    try {
        $ErrorActionPreference = "Continue"
        $global:LASTEXITCODE = $null
        & $Executable @ArgumentList
        $nativeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }

    if ($null -eq $nativeExitCode) {
        [Console]::Error.WriteLine("$FailureMessage (native process did not report an exit code)")
        exit 1
    }
    if ($nativeExitCode -ne 0) {
        [Console]::Error.WriteLine("$FailureMessage (exit code: $nativeExitCode)")
        exit $nativeExitCode
    }
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    [Console]::Error.WriteLine("GeniVox virtual-environment Python was not found: $venvPython")
    [Console]::Error.WriteLine("Run scripts\bootstrap_windows.ps1 first.")
    exit 1
}

$hadPythonUtf8 = Test-Path Env:PYTHONUTF8
$hadQtPlatform = Test-Path Env:QT_QPA_PLATFORM
$previousPythonUtf8 = $null
$previousQtPlatform = $null
if ($hadPythonUtf8) { $previousPythonUtf8 = $env:PYTHONUTF8 }
if ($hadQtPlatform) { $previousQtPlatform = $env:QT_QPA_PLATFORM }

try {
    $env:PYTHONUTF8 = "1"
    $env:QT_QPA_PLATFORM = "offscreen"

    Invoke-NativeCommand -Executable $venvPython `
        -ArgumentList @("-m", "pip", "--disable-pip-version-check", "check") `
        -FailureMessage "Installed Python packages are inconsistent."

    Invoke-NativeCommand -Executable $venvPython `
        -ArgumentList @("-m", "genivox.selftest") `
        -FailureMessage "GeniVox self-test failed."
} finally {
    if ($hadPythonUtf8) {
        $env:PYTHONUTF8 = $previousPythonUtf8
    } else {
        Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
    }
    if ($hadQtPlatform) {
        $env:QT_QPA_PLATFORM = $previousQtPlatform
    } else {
        Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
    }
}

Write-Host "BASE_VERIFIED"
