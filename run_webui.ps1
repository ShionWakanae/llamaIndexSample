Set-Location $PSScriptRoot
Write-Host "Current Dir:" (Get-Location)

.\venv\Scripts\Activate.ps1

python .\src\rag_webui.py
pause