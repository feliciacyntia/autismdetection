# Jalankan / lanjutkan runner modeling 06 (proses mandiri; tidak bergantung pada sesi chat).
# Pakai:
#   powershell -ExecutionPolicy Bypass -File .\run_pilot.ps1                      (pilot v2, lanjut otomatis)
#   powershell -ExecutionPolicy Bypass -File .\run_pilot.ps1 -StopAt 07:00        (berhenti rapi pukul 07:00)
#   powershell -ExecutionPolicy Bypass -File .\run_pilot.ps1 -RetryFailed         (ulang unit yang gagal)
#   powershell -ExecutionPolicy Bypass -File .\run_pilot.ps1 -Version v1          (runner pilot v1)
param(
    [ValidateSet("pilot200", "all")] [string]$Subset = "pilot200",
    [ValidateSet("v1", "v2")] [string]$Version = "v2",
    [string]$StopAt = "",
    [switch]$RetryFailed
)
$py = Join-Path $PSScriptRoot ".venv-model\Scripts\python.exe"
$dirName = if ($Version -eq "v1") { $Subset } else { "${Subset}_$Version" }
$runner = if ($Version -eq "v1") { "src\run_pilot.py" } else { "src\run_pilot_v2.py" }
$tables = if ($Version -eq "v1") { "src\make_tables.py" } else { "src\make_tables_v2.py" }
$stop = "C:\ta_local\results\$dirName\STOP"
Set-Location $PSScriptRoot
Remove-Item $stop -ErrorAction SilentlyContinue
$ram = [double](& $py -c "import psutil; print(psutil.virtual_memory().available / 2**30)")
if ($ram -lt 4) {
    Write-Warning ("RAM tersedia {0:N2} GB < 4 GB. Tutup aplikasi lain lalu jalankan lagi." -f $ram)
    exit 2
}
Write-Host ("RAM tersedia {0:N2} GB. Menjalankan runner {1} ({2})..." -f $ram, $Subset, $Version)
$env:PYTHONIOENCODING = "utf-8"
$argv = @($runner, "--subset", $Subset, "--resume")
if ($StopAt) { $argv += @("--stop-at", $StopAt) }
if ($RetryFailed) { $argv += "--retry-failed" }
& $py @argv
$code = $LASTEXITCODE
Write-Host "EXIT CODE: $code  (0 = semua selesai, 1 = ada unit gagal/dilewati, 2 = tidak dimulai, 3 = dihentikan rapi)"
& $py $tables --subset $Subset
exit $code
