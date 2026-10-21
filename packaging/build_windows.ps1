# Build the Windows folder distribution and check it: dist\DefectLab\
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 [-Python .venv\Scripts\python.exe]
# Needs: pip install -e ".[package]" in that environment (PyInstaller).
param([string]$Python = ".venv\Scripts\python.exe")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONUTF8 = "1"

& $Python -m PyInstaller --version | Out-Null
if ($LASTEXITCODE -ne 0) { throw "PyInstaller가 없습니다: $Python -m pip install -e '.[package]'" }
& $Python -m PyInstaller --noconfirm --clean --distpath dist --workpath build packaging\defectlab_windows.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 빌드 실패" }

Write-Host "== self-check (설치본, 콘솔 worker로 실행) =="
$env:QT_QPA_PLATFORM = "offscreen"
& dist\DefectLab\DefectLab-worker.exe --help | Out-Null
if ($LASTEXITCODE -ne 0) { throw "DefectLab-worker.exe 실행 실패" }
# the windowed exe has no console: run its self-check through cmd to capture the output
$out = cmd /c "dist\DefectLab\DefectLab.exe --self-check 2>&1"
$code = $LASTEXITCODE
$out | Write-Host
Remove-Item Env:QT_QPA_PLATFORM
if ($code -ne 0) { throw "self-check 실패" }

Compress-Archive -Force -Path dist\DefectLab -DestinationPath dist\DefectLab-windows-x64.zip
Write-Host "완료: dist\DefectLab\DefectLab.exe, dist\DefectLab-windows-x64.zip"
