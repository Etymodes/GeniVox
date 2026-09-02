param(
    [switch]$WithDev
)

$ErrorActionPreference = "Stop"

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$ArgumentList = @(),
        [Parameter(Mandatory = $true)][string]$FailureMessage
    )

    # Windows PowerShell 5.1 can promote a native program's harmless stderr
    # output to a terminating ErrorRecord when ErrorActionPreference is Stop.
    $previousPreference = $ErrorActionPreference
    $nativeExitCode = $null
    try {
        $ErrorActionPreference = "Continue"
        & $Executable @ArgumentList
        $nativeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($null -eq $nativeExitCode -or $nativeExitCode -ne 0) {
        throw "$FailureMessage (exit code: $nativeExitCode)"
    }
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    $pyLauncher = Get-Command py -CommandType Application -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        Invoke-NativeCommand -Executable $pyLauncher.Source `
            -ArgumentList @("-3.11", "-m", "venv", $venvPath) `
            -FailureMessage "Could not create a Python 3.11 environment. Install Python 3.11 and retry."
    } else {
        $python = Get-Command python -ErrorAction Stop
        $version = Invoke-NativeCommand -Executable $python.Source `
            -ArgumentList @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") `
            -FailureMessage "Could not inspect the Python version."
        if ($version -ne "3.11") {
            throw "GeniVox requires Python 3.11. Found Python $version and no Windows py launcher."
        }
        Invoke-NativeCommand -Executable $python.Source `
            -ArgumentList @("-m", "venv", $venvPath) `
            -FailureMessage "Could not create the GeniVox virtual environment."
    }
}

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment creation did not produce $venvPython."
}

Invoke-NativeCommand -Executable $venvPython `
    -ArgumentList @("-c", "import struct,sys; assert sys.version_info[:2] == (3, 11), sys.version; assert struct.calcsize('P') == 8, '64-bit Python required'") `
    -FailureMessage "The existing .venv must be a healthy 64-bit Python 3.11 environment. Remove it and retry."

Invoke-NativeCommand -Executable $venvPython `
    -ArgumentList @("-m", "pip", "install", "--upgrade", "pip") `
    -FailureMessage "pip upgrade failed."

$extras = @()
if ($WithDev) { $extras += "dev" }
$target = $projectRoot
if ($extras.Count -gt 0) {
    $target = "$projectRoot[$($extras -join ',')]"
}

Invoke-NativeCommand -Executable $venvPython `
    -ArgumentList @("-m", "pip", "install", "-e", $target) `
    -FailureMessage "GeniVox installation failed."

Write-Host "GeniVox environment is ready: $venvPath"
Write-Host "No model weights were downloaded. Register local engines from Model Manager."
