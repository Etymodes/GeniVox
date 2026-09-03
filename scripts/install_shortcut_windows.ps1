param(
    [string]$ShortcutPath = (Join-Path `
        ([Environment]::GetFolderPath([Environment+SpecialFolder]::DesktopDirectory)) `
        "GeniVox.lnk")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

$projectRoot = Split-Path -Parent $PSScriptRoot
$runScript = Join-Path $PSScriptRoot "run_windows.ps1"
$iconPath = Join-Path $projectRoot "src\genivox\assets\genivox-app-icon.ico"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$windowsPowerShell = Join-Path $PSHOME "powershell.exe"
$shortcutDirectory = Split-Path -Parent $ShortcutPath
$expectedArguments = "-NoLogo -NoProfile -ExecutionPolicy Bypass -File `"$runScript`""

foreach ($requiredFile in @($runScript, $iconPath, $venvPython, $windowsPowerShell)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required file not found: $requiredFile"
    }
}
if (-not (Test-Path -LiteralPath $shortcutDirectory -PathType Container)) {
    throw "Shortcut directory not found: $shortcutDirectory"
}

$temporaryShortcutPath = Join-Path `
    $shortcutDirectory `
    (".GeniVox-{0}.tmp.lnk" -f [Guid]::NewGuid().ToString("N"))
$shell = $null
$shortcut = $null
try {
    try {
        $shell = New-Object -ComObject WScript.Shell
        $shortcut = $shell.CreateShortcut($temporaryShortcutPath)
        $shortcut.TargetPath = $windowsPowerShell
        $shortcut.Arguments = $expectedArguments
        $shortcut.WorkingDirectory = $projectRoot
        $shortcut.IconLocation = "$iconPath,0"
        $shortcut.Description = "GeniVox local multilingual voice laboratory"
        $shortcut.Hotkey = ""
        $shortcut.WindowStyle = 7
        $shortcut.Save()
    } finally {
        if ($null -ne $shortcut -and [Runtime.InteropServices.Marshal]::IsComObject($shortcut)) {
            [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($shortcut)
        }
        if ($null -ne $shell -and [Runtime.InteropServices.Marshal]::IsComObject($shell)) {
            [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($shell)
        }
        $shortcut = $null
        $shell = $null
    }

    if (-not (Test-Path -LiteralPath $temporaryShortcutPath -PathType Leaf)) {
        throw "Shortcut creation did not produce: $temporaryShortcutPath"
    }
    if (Test-Path -LiteralPath $ShortcutPath) {
        $existingShortcut = Get-Item -LiteralPath $ShortcutPath -Force
        if ($existingShortcut.PSIsContainer) {
            throw "Shortcut destination is a directory: $ShortcutPath"
        }
        if (($existingShortcut.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Shortcut destination is a reparse point: $ShortcutPath"
        }
        [IO.File]::Replace($temporaryShortcutPath, $ShortcutPath, $null)
    } else {
        [IO.File]::Move($temporaryShortcutPath, $ShortcutPath)
    }
} finally {
    if (Test-Path -LiteralPath $temporaryShortcutPath -PathType Leaf) {
        Remove-Item -LiteralPath $temporaryShortcutPath -Force
    }
}

if (-not (Test-Path -LiteralPath $ShortcutPath -PathType Leaf)) {
    throw "Shortcut installation did not produce: $ShortcutPath"
}

$shell = $null
$shortcut = $null
try {
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($ShortcutPath)
    if ($shortcut.TargetPath -ne $windowsPowerShell -or
        $shortcut.Arguments -ne $expectedArguments -or
        $shortcut.WorkingDirectory -ne $projectRoot -or
        $shortcut.IconLocation -ne "$iconPath,0" -or
        $shortcut.WindowStyle -ne 7 -or
        $shortcut.Hotkey) {
        throw "Shortcut verification failed: $ShortcutPath"
    }
} finally {
    if ($null -ne $shortcut -and [Runtime.InteropServices.Marshal]::IsComObject($shortcut)) {
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($shortcut)
    }
    if ($null -ne $shell -and [Runtime.InteropServices.Marshal]::IsComObject($shell)) {
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($shell)
    }
}

Write-Host "SHORTCUT_READY"
Write-Host "GeniVox desktop shortcut is ready: $ShortcutPath"
