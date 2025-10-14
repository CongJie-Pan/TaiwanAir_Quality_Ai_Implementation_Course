# PowerShell Script to Fix Windows Firewall for Streamlit
# Must run as Administrator
# Author: Claude Code
# Date: 2025-10-14

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "ERROR: Administrator privileges required" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run PowerShell as Administrator:" -ForegroundColor Yellow
    Write-Host "1. Search 'PowerShell' in Start Menu" -ForegroundColor White
    Write-Host "2. Right-click -> Run as Administrator" -ForegroundColor White
    Write-Host "3. Run this script again" -ForegroundColor White
    Write-Host ""
    pause
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Fix Windows Firewall for Streamlit" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get Python path from virtual environment
$pythonPath = "$PSScriptRoot\.venv\Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
    Write-Host "❌ Python not found in virtual environment!" -ForegroundColor Red
    Write-Host "   Expected path: $pythonPath" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please check your virtual environment setup." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "✅ Found Python: $pythonPath" -ForegroundColor Green
Write-Host ""

# Add firewall rules
Write-Host "[Step 1] Adding Windows Firewall rules..." -ForegroundColor Yellow
Write-Host ""

try {
    # Inbound rule
    Write-Host "Adding inbound rule..." -ForegroundColor Cyan
    New-NetFirewallRule -DisplayName "Streamlit Python (Inbound)" `
                        -Direction Inbound `
                        -Program $pythonPath `
                        -Action Allow `
                        -Protocol TCP `
                        -LocalPort 8501-8510 `
                        -Profile Any `
                        -ErrorAction Stop | Out-Null

    Write-Host "✅ Inbound rule added" -ForegroundColor Green

    # Outbound rule
    Write-Host "Adding outbound rule..." -ForegroundColor Cyan
    New-NetFirewallRule -DisplayName "Streamlit Python (Outbound)" `
                        -Direction Outbound `
                        -Program $pythonPath `
                        -Action Allow `
                        -Protocol TCP `
                        -Profile Any `
                        -ErrorAction Stop | Out-Null

    Write-Host "✅ Outbound rule added" -ForegroundColor Green
    Write-Host ""
    Write-Host "✅ Firewall rules successfully configured!" -ForegroundColor Green

} catch {
    if ($_.Exception.Message -like "*already exists*") {
        Write-Host "⚠️  Firewall rules already exist, removing and recreating..." -ForegroundColor Yellow

        # Remove existing rules
        Remove-NetFirewallRule -DisplayName "Streamlit Python (Inbound)" -ErrorAction SilentlyContinue
        Remove-NetFirewallRule -DisplayName "Streamlit Python (Outbound)" -ErrorAction SilentlyContinue

        # Recreate rules
        New-NetFirewallRule -DisplayName "Streamlit Python (Inbound)" `
                            -Direction Inbound `
                            -Program $pythonPath `
                            -Action Allow `
                            -Protocol TCP `
                            -LocalPort 8501-8510 `
                            -Profile Any | Out-Null

        New-NetFirewallRule -DisplayName "Streamlit Python (Outbound)" `
                            -Direction Outbound `
                            -Program $pythonPath `
                            -Action Allow `
                            -Protocol TCP `
                            -Profile Any | Out-Null

        Write-Host "✅ Firewall rules recreated successfully!" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to add firewall rules" -ForegroundColor Red
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "[Step 2] Checking reserved port ranges..." -ForegroundColor Yellow
Write-Host ""

# Check for Hyper-V reserved ports
$reservedPorts = netsh interface ipv4 show excludedportrange protocol=tcp

if ($reservedPorts -match "8501|8502|8503|8504|8505") {
    Write-Host "⚠️  Warning: Some ports in range 8501-8510 are reserved by Hyper-V or Windows" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Reserved port ranges:" -ForegroundColor Cyan
    Write-Host $reservedPorts
    Write-Host ""
    Write-Host "💡 Suggestion: Try using port 9000 or higher" -ForegroundColor Cyan
    Write-Host "   streamlit run src/main/python/app.py --server.port 9000" -ForegroundColor White
} else {
    Write-Host "✅ Ports 8501-8510 are not reserved" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Configuration Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Now try running Streamlit:" -ForegroundColor Green
Write-Host "   streamlit run src/main/python/app.py" -ForegroundColor White
Write-Host ""
Write-Host "Or with a different port:" -ForegroundColor Green
Write-Host "   streamlit run src/main/python/app.py --server.port 9000" -ForegroundColor White
Write-Host ""

pause
