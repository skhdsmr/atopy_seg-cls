# ============================================================
# DermaScan local-network setup (Windows -> WSL port forwarding + firewall)
# Just run it; it will request Administrator rights (UAC) automatically.
#   powershell -ExecutionPolicy Bypass -File c:\hsbioMVP\server\setup_local.ps1
# Re-run after "wsl --shutdown" or a PC reboot (WSL IP may change).
# ============================================================

# --- self-elevate to Administrator ---
$isAdmin = ([Security.Principal.WindowsPrincipal] `
  [Security.Principal.WindowsIdentity]::GetCurrent()
  ).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)

if (-not $isAdmin) {
  Write-Host "Requesting Administrator rights (UAC prompt)..." -ForegroundColor Yellow
  Start-Process powershell -Verb RunAs -ArgumentList `
    "-NoExit -ExecutionPolicy Bypass -File `"$PSCommandPath`""
  exit
}

$port = 8000

# 1) WSL IP (dynamic)
$wslIp = (wsl hostname -I).Trim().Split(" ")[0]
if (-not $wslIp) { Write-Host "WSL IP not found. Is WSL running?" -ForegroundColor Red; exit 1 }
Write-Host "WSL IP      : $wslIp"

# 2) port forwarding 0.0.0.0:8000 -> WSL:8000
netsh interface portproxy delete v4tov4 listenport=$port listenaddress=0.0.0.0 2>$null | Out-Null
netsh interface portproxy add v4tov4 listenport=$port listenaddress=0.0.0.0 connectport=$port connectaddress=$wslIp | Out-Null
Write-Host "Port forward: 0.0.0.0:$port -> ${wslIp}:$port"

# 3) firewall inbound allow
Remove-NetFirewallRule -DisplayName "DermaScan WSL $port" -ErrorAction SilentlyContinue
New-NetFirewallRule -DisplayName "DermaScan WSL $port" -Direction Inbound -Action Allow `
  -Protocol TCP -LocalPort $port -Profile Any | Out-Null
Write-Host "Firewall    : allow inbound TCP $port"

# 4) PC Wi-Fi IP for the app BASE_URL
$wifiIp = (Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object { $_.InterfaceAlias -match "Wi-Fi" -and $_.IPAddress -notlike "169.*" } |
  Select-Object -First 1).IPAddress

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  PC Wi-Fi IP : $wifiIp" -ForegroundColor Green
Write-Host "  app BASE_URL: http://${wifiIp}:$port" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Checklist:"
Write-Host "  1) WSL server running (uvicorn ... --host 0.0.0.0 --port $port)"
Write-Host "  2) phone and PC on the SAME network (hotspot / Wi-Fi)"
Write-Host "  3) client.js BASE_URL = http://${wifiIp}:$port"
Write-Host "  4) restart app: npx expo start --tunnel --clear"
Write-Host ""
Write-Host "Current portproxy rules:"
netsh interface portproxy show v4tov4
Write-Host ""
Write-Host "Self-test (PC -> WSL via forward):"
try {
  $r = Invoke-WebRequest -Uri "http://${wifiIp}:$port/health" -TimeoutSec 8 -UseBasicParsing
  Write-Host ("  http://{0}:{1}/health -> {2} {3}" -f $wifiIp,$port,$r.StatusCode,$r.Content) -ForegroundColor Green
} catch {
  Write-Host "  health check failed: $($_.Exception.Message)" -ForegroundColor Red
  Write-Host "  (is the WSL server running on :$port ?)"
}
