# Lihat progres runner 06 (hanya membaca; aman kapan saja, juga saat runner berjalan).
# Pakai: powershell -ExecutionPolicy Bypass -File .\progress_pilot.ps1 [-Version v2] [-Subset pilot200] [-Tables]
param([ValidateSet("pilot200", "all")] [string]$Subset = "pilot200", [ValidateSet("v1", "v2")] [string]$Version = "v2",
      [switch]$Tables)
$py = Join-Path $PSScriptRoot ".venv-model\Scripts\python.exe"
$dirName = if ($Version -eq "v1") { $Subset } else { "${Subset}_$Version" }
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
& $py "src\pilot_progress.py" --results-dir "C:\ta_local\results\$dirName"
if ($Tables) {
    $tables = if ($Version -eq "v1") { "src\make_tables.py" } else { "src\make_tables_v2.py" }
    & $py $tables --subset $Subset
}
