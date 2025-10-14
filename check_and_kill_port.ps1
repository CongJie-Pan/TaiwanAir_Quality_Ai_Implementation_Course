# PowerShell Script to Check and Kill Process Using Port 8501
# Author: Claude Code
# Date: 2025-10-14

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Checking Port 8501 Usage" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if port 8501 is being used
Write-Host "[Step 1] Checking port 8501..." -ForegroundColor Yellow

$connections = Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue

if ($null -eq $connections) {
    Write-Host ""
    Write-Host "✅ Port 8501 is available!" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now run Streamlit:" -ForegroundColor Cyan
    Write-Host "   streamlit run src/main/python/app.py" -ForegroundColor White
    Write-Host ""
    pause
    exit 0
}

Write-Host ""
Write-Host "⚠️  Port 8501 is being used!" -ForegroundColor Red
Write-Host ""

# Step 2: Get the PID
$PID_Using_Port = $connections[0].OwningProcess

Write-Host "[Step 2] Process PID using port: $PID_Using_Port" -ForegroundColor Yellow
Write-Host ""

# Step 3: Get process information
Write-Host "[Step 3] Getting process details..." -ForegroundColor Yellow

$process = Get-Process -Id $PID_Using_Port -ErrorAction SilentlyContinue

if ($null -ne $process) {
    Write-Host ""
    Write-Host "Process Name: " -NoNewline -ForegroundColor Cyan
    Write-Host $process.ProcessName -ForegroundColor White

    Write-Host "Process ID:   " -NoNewline -ForegroundColor Cyan
    Write-Host $PID_Using_Port -ForegroundColor White

    Write-Host "Start Time:   " -NoNewline -ForegroundColor Cyan
    Write-Host $process.StartTime -ForegroundColor White

    if ($process.Path) {
        Write-Host "Path:         " -NoNewline -ForegroundColor Cyan
        Write-Host $process.Path -ForegroundColor White
    }

    Write-Host "CPU Time:     " -NoNewline -ForegroundColor Cyan
    Write-Host "$($process.CPU) seconds" -ForegroundColor White

    Write-Host "Memory:       " -NoNewline -ForegroundColor Cyan
    Write-Host "$([math]::Round($process.WorkingSet64/1MB, 2)) MB" -ForegroundColor White
} else {
    Write-Host "Unable to get process details" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Do you want to terminate this process?" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "⚠️  WARNING: Please confirm the process name above" -ForegroundColor Yellow
Write-Host "   Do NOT terminate if it's a critical system service!" -ForegroundColor Yellow
Write-Host ""

# Step 4: Confirm and kill
$confirmation = Read-Host "Confirm to terminate process $PID_Using_Port? (Y/N)"

if ($confirmation -ne 'Y' -and $confirmation -ne 'y') {
    Write-Host ""
    Write-Host "❌ Operation cancelled" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Suggestion: Use a different port to run Streamlit" -ForegroundColor Cyan
    Write-Host "   streamlit run src/main/python/app.py --server.port 8502" -ForegroundColor White
    Write-Host ""
    pause
    exit 1
}

Write-Host ""
Write-Host "[Step 4] Terminating process $PID_Using_Port..." -ForegroundColor Yellow

try {
    Stop-Process -Id $PID_Using_Port -Force -ErrorAction Stop
    Write-Host ""
    Write-Host "✅ Process terminated successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now run Streamlit:" -ForegroundColor Cyan
    Write-Host "   streamlit run src/main/python/app.py" -ForegroundColor White
    Write-Host ""
} catch {
    Write-Host ""
    Write-Host "❌ Failed to terminate process! May need administrator privileges" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error message: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Solutions:" -ForegroundColor Cyan
    Write-Host "   1. Run PowerShell as Administrator" -ForegroundColor White
    Write-Host "   2. Or use a different port: streamlit run src/main/python/app.py --server.port 8502" -ForegroundColor White
    Write-Host ""
}

pause
