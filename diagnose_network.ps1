# PowerShell Script to Diagnose Network Permission Issues
# Author: Claude Code
# Date: 2025-10-14

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Network Permission Diagnostic Tool" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check 1: Administrator privileges
Write-Host "[Check 1] Administrator Privileges" -ForegroundColor Yellow
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if ($isAdmin) {
    Write-Host "✅ Running as Administrator" -ForegroundColor Green
} else {
    Write-Host "❌ NOT running as Administrator" -ForegroundColor Red
    Write-Host "   This is likely the cause of the permission error!" -ForegroundColor Yellow
}
Write-Host ""

# Check 2: Firewall status
Write-Host "[Check 2] Windows Firewall Status" -ForegroundColor Yellow
$firewallProfiles = Get-NetFirewallProfile
foreach ($profile in $firewallProfiles) {
    $status = if ($profile.Enabled) { "ON" } else { "OFF" }
    $color = if ($profile.Enabled) { "Yellow" } else { "Green" }
    Write-Host "   $($profile.Name): $status" -ForegroundColor $color
}
Write-Host ""

# Check 3: Check for existing firewall rules
Write-Host "[Check 3] Existing Firewall Rules for Python" -ForegroundColor Yellow
$pythonRules = Get-NetFirewallRule | Where-Object { $_.DisplayName -like "*Python*" -or $_.DisplayName -like "*Streamlit*" }

if ($pythonRules.Count -gt 0) {
    Write-Host "✅ Found $($pythonRules.Count) existing Python/Streamlit firewall rules:" -ForegroundColor Green
    foreach ($rule in $pythonRules) {
        Write-Host "   - $($rule.DisplayName) [$($rule.Direction)] [$($rule.Action)]" -ForegroundColor White
    }
} else {
    Write-Host "❌ No Python/Streamlit firewall rules found" -ForegroundColor Red
    Write-Host "   You may need to add firewall rules" -ForegroundColor Yellow
}
Write-Host ""

# Check 4: Hyper-V reserved ports
Write-Host "[Check 4] Checking for Hyper-V Reserved Ports" -ForegroundColor Yellow
Write-Host "   Querying reserved port ranges..." -ForegroundColor Cyan

try {
    $reservedPortsOutput = netsh interface ipv4 show excludedportrange protocol=tcp 2>&1

    # Parse the output to find port ranges
    $portRanges = $reservedPortsOutput | Select-String -Pattern "\s+(\d+)\s+(\d+)" | ForEach-Object {
        if ($_.Matches.Groups[1].Value -and $_.Matches.Groups[2].Value) {
            [PSCustomObject]@{
                Start = [int]$_.Matches.Groups[1].Value
                End = [int]$_.Matches.Groups[2].Value
            }
        }
    }

    # Check if Streamlit ports are in reserved ranges
    $streamlitPorts = 8501..8510
    $conflictFound = $false

    foreach ($port in $streamlitPorts) {
        foreach ($range in $portRanges) {
            if ($port -ge $range.Start -and $port -le $range.End) {
                if (-not $conflictFound) {
                    Write-Host "⚠️  Warning: Some Streamlit ports are reserved:" -ForegroundColor Yellow
                    $conflictFound = $true
                }
                Write-Host "   Port $port is reserved (range: $($range.Start)-$($range.End))" -ForegroundColor Red
            }
        }
    }

    if (-not $conflictFound) {
        Write-Host "✅ Ports 8501-8510 are NOT in reserved ranges" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "💡 Try using ports outside reserved ranges:" -ForegroundColor Cyan
        Write-Host "   Port 9000, 9001, 9002, etc." -ForegroundColor White
    }
} catch {
    Write-Host "⚠️  Could not check reserved ports" -ForegroundColor Yellow
}
Write-Host ""

# Check 5: Antivirus software
Write-Host "[Check 5] Antivirus Software Detection" -ForegroundColor Yellow
$antivirusProducts = Get-CimInstance -Namespace root/SecurityCenter2 -ClassName AntivirusProduct -ErrorAction SilentlyContinue

if ($antivirusProducts) {
    Write-Host "⚠️  Detected antivirus software:" -ForegroundColor Yellow
    foreach ($av in $antivirusProducts) {
        Write-Host "   - $($av.displayName)" -ForegroundColor White
    }
    Write-Host "   These may block network access for Python" -ForegroundColor Yellow
} else {
    Write-Host "ℹ️  No third-party antivirus detected (or cannot query)" -ForegroundColor Cyan
}
Write-Host ""

# Check 6: Network adapter status
Write-Host "[Check 6] Network Adapter Status" -ForegroundColor Yellow
$adapters = Get-NetAdapter | Where-Object { $_.Status -eq "Up" }
Write-Host "✅ Active network adapters: $($adapters.Count)" -ForegroundColor Green
foreach ($adapter in $adapters) {
    Write-Host "   - $($adapter.Name) [$($adapter.InterfaceDescription)]" -ForegroundColor White
}
Write-Host ""

# Check 7: Test simple socket binding
Write-Host "[Check 7] Test Socket Binding (Port 9999)" -ForegroundColor Yellow
try {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, 9999)
    $listener.Start()
    $listener.Stop()
    Write-Host "✅ Can bind to network sockets" -ForegroundColor Green
} catch {
    Write-Host "❌ CANNOT bind to network sockets!" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "   This confirms a permission/firewall issue" -ForegroundColor Yellow
}
Write-Host ""

# Summary and recommendations
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Diagnosis Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "📋 Recommendations:" -ForegroundColor Cyan
Write-Host ""

if (-not $isAdmin) {
    Write-Host "🔴 CRITICAL: Run as Administrator" -ForegroundColor Red
    Write-Host "   1. Close this window" -ForegroundColor White
    Write-Host "   2. Right-click PowerShell -> Run as Administrator" -ForegroundColor White
    Write-Host "   3. Try running Streamlit again" -ForegroundColor White
    Write-Host ""
}

if ($pythonRules.Count -eq 0) {
    Write-Host "🟡 Add Firewall Rules:" -ForegroundColor Yellow
    Write-Host "   Run: .\fix_firewall.ps1 (as Administrator)" -ForegroundColor White
    Write-Host ""
}

Write-Host "🟢 Alternative Solutions:" -ForegroundColor Green
Write-Host "   1. Use a high port number (>9000):" -ForegroundColor White
Write-Host "      streamlit run src/main/python/app.py --server.port 9000" -ForegroundColor White
Write-Host ""
Write-Host "   2. Temporarily disable Windows Firewall (not recommended):" -ForegroundColor White
Write-Host "      - Windows Settings -> Windows Security -> Firewall" -ForegroundColor White
Write-Host ""
Write-Host "   3. Check your antivirus settings" -ForegroundColor White
Write-Host "      - Add Python.exe to the exclusion list" -ForegroundColor White
Write-Host ""

pause
