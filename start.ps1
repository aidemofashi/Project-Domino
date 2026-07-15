param([string]$Device = "cpu")

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = "$Root\.venv\Scripts\python.exe"
$TauriExe = "$Root\Tilps\Ui\tauri\src-tauri\target\release\domino-ui.exe"

$env:DEVICE = $Device

Write-Host "[Domino] Starting backend..." -ForegroundColor Green
$pyProc = Start-Process -FilePath $VenvPython -ArgumentList "$Root\start.py" -NoNewWindow -PassThru

Start-Sleep -Seconds 3

if (Test-Path $TauriExe) {
    Write-Host "[Domino] Starting UI..." -ForegroundColor Green
    $uiProc = Start-Process -FilePath $TauriExe -Wait -PassThru
    Write-Host "[Domino] UI closed" -ForegroundColor Yellow
} else {
    Write-Host "[Domino] UI binary not found. Build it first:" -ForegroundColor Red
    Write-Host "  cd ui; npm run tauri build" -ForegroundColor Yellow
}

Write-Host "[Domino] Stopping backend..." -ForegroundColor Green
if ($pyProc -and !$pyProc.HasExited) {
    $pyProc.Kill()
}
Write-Host "[Domino] Exited" -ForegroundColor Green
