$pythonw = where.exe pythonw 2>$null | Select-Object -First 1
if ($null -eq $pythonw) {
    Write-Host "pythonw.exe not found in PATH!"
    exit 1
}
& $pythonw "$PSScriptRoot\app.py"
