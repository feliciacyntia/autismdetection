# Resume batch 05c: hapus STOP, cek RAM >= 4 GB, lalu lanjutkan dari checkpoint.
# Pakai:
#   powershell -ExecutionPolicy Bypass -File .\resume_05c.ps1                          (subset200)
#   powershell -ExecutionPolicy Bypass -File .\resume_05c.ps1 -Mode all -StopAt 07:00  (batch malam, pause otomatis 07:00)
param(
    [ValidateSet("subset200", "all")] [string]$Mode = "subset200",
    [string]$StopAt = ""
)
$py = "C:\Program Files\Python312\python.exe"
$stop = "C:\ta_work\05c\STOP"
Set-Location $PSScriptRoot
Remove-Item $stop -ErrorAction SilentlyContinue
$ram = [double](& $py -c "import psutil; print(psutil.virtual_memory().available / 2**30)")
if ($ram -lt 4) {
    Write-Warning ("RAM tersedia {0:N2} GB < 4 GB. Tutup aplikasi lain (VS Code, Chrome, dll.) lalu jalankan lagi." -f $ram)
    exit 2
}
if ($StopAt) {
    # buat file STOP pada jam tertentu; batch menyelesaikan subjek yang berjalan lalu berhenti rapi
    $t = [datetime]::ParseExact($StopAt, "HH:mm", $null)
    if ($t -le (Get-Date)) { $t = $t.AddDays(1) }
    Start-Job -ScriptBlock { param($when, $file) Start-Sleep -Seconds ([int]($when - (Get-Date)).TotalSeconds); New-Item -ItemType File -Force $file | Out-Null } -ArgumentList $t, $stop | Out-Null
    Write-Host ("Pause otomatis dijadwalkan pukul {0:HH:mm} ({0:dd MMM})." -f $t)
}
Write-Host ("RAM tersedia {0:N2} GB. Melanjutkan batch {1}..." -f $ram, $Mode)
$env:PYTHONIOENCODING = "utf-8"
Remove-Item Env:ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS -ErrorAction SilentlyContinue
& $py "src\run_05c.py" --mode $Mode --resume --clean-work --output-root "C:\ta_local"
$code = $LASTEXITCODE
Get-Job | Remove-Job -Force -ErrorAction SilentlyContinue     # batalkan jadwal pause jika batch selesai lebih dulu
Write-Host "EXIT CODE: $code  (0 = tuntas, 1 = ada subjek gagal, 2 = tidak dimulai, 3 = di-pause/dihentikan)"
exit $code
