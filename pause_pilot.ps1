# Hentikan runner 06 dengan rapi: epoch yang berjalan diselesaikan, checkpoint disimpan, lalu keluar.
# Pakai: powershell -ExecutionPolicy Bypass -File .\pause_pilot.ps1 [-Version v2] [-Subset pilot200]
param([ValidateSet("pilot200", "all")] [string]$Subset = "pilot200", [ValidateSet("v1", "v2")] [string]$Version = "v2")
$dirName = if ($Version -eq "v1") { $Subset } else { "${Subset}_$Version" }
$dir = "C:\ta_local\results\$dirName"
New-Item -ItemType File -Force (Join-Path $dir "STOP") | Out-Null
Write-Host "STOP dibuat di $dir. Menunggu epoch yang berjalan selesai..."
$deadline = (Get-Date).AddMinutes(15)
while ((Test-Path (Join-Path $dir "run.lock")) -and ((Get-Date) -lt $deadline)) { Start-Sleep -Seconds 5 }
if (Test-Path (Join-Path $dir "run.lock")) { Write-Warning "Runner belum berhenti setelah 15 menit; cek jendelanya."; exit 1 }
Write-Host "Runner sudah berhenti. Lanjutkan nanti dengan run_pilot.ps1"
