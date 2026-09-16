param([string]$Device = "cpu")

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = "$Root\.venv\Scripts\python.exe"

$env:DEVICE = $Device

# UI / WebSocket 是否启动由 Data/api.json 的 system 配置控制 (enable_ui / enable_ws)
Write-Host "[Domino] Starting backend (press Ctrl+C to exit)..." -ForegroundColor Green
& $VenvPython "$Root\start.py"
Write-Host "[Domino] Exited" -ForegroundColor Green
