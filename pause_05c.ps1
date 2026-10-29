# Pause batch 05c: buat file STOP. Batch menyelesaikan subjek yang sedang berjalan, lalu keluar rapi.
# Pakai: powershell -ExecutionPolicy Bypass -File .\pause_05c.ps1
$stop = "C:\ta_work\05c\STOP"
$lock = "C:\ta_work\05c\run.lock"
New-Item -ItemType File -Force $stop | Out-Null
Write-Host "STOP dibuat: $stop"
if (-not (Test-Path $lock)) { Write-Host "Tidak ada batch yang berjalan. Aman untuk sleep."; exit 0 }
Write-Host "Menunggu subjek yang sedang berjalan selesai (biasanya < 3 menit)..."
$deadline = (Get-Date).AddMinutes(15)
while ((Test-Path $lock) -and ((Get-Date) -lt $deadline)) { Start-Sleep -Seconds 5 }
if (Test-Path $lock) {
    Write-Warning "Batch belum berhenti setelah 15 menit. Cek jendela batch sebelum sleep."
    exit 1
}
Write-Host "Batch sudah berhenti. Aman untuk sleep. Lanjutkan nanti dengan resume_05c.ps1"
