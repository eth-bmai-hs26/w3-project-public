# ============================================
# Python Installer Script for Windows
# Run as Administrator for best results
# ============================================

$PythonVersion = "3.12.9"
$InstallerUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
$InstallerPath = "$env:TEMP\python-installer.exe"

Write-Host "==============================" -ForegroundColor Cyan
Write-Host "  Python $PythonVersion Installer" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan

# Check if Python is already installed
Write-Host "`nChecking if Python is already installed..." -ForegroundColor Yellow
$existingPython = Get-Command python -ErrorAction SilentlyContinue
if ($existingPython) {
    $existingVersion = & python --version
    Write-Host "Python is already installed: $existingVersion" -ForegroundColor Green    
} else {
    Write-Host "Python not found. Proceeding with installation..." -ForegroundColor Yellow

    # Download Python installer
    Write-Host "`nDownloading Python $PythonVersion..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $InstallerUrl -OutFile $InstallerPath
    Write-Host "Download complete!" -ForegroundColor Green

    # Install Python silently with PATH and pip included
    Write-Host "`nInstalling Python..." -ForegroundColor Yellow
    Start-Process -FilePath $InstallerPath -ArgumentList `
        "/quiet", `
        "InstallAllUsers=1", `
        "PrependPath=1", `
        "Include_pip=1", `
        "Include_launcher=1" `
        -Wait

    Write-Host "Installation complete!" -ForegroundColor Green

    # Cleanup installer
    Remove-Item $InstallerPath -Force
    Write-Host "Cleaned up installer." -ForegroundColor Gray

    # Verify installation
    Write-Host "`nVerifying installation..." -ForegroundColor Yellow
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

    $pythonPath = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonPath) {
        $version = & python --version
        Write-Host "SUCCESS: $version is installed!" -ForegroundColor Green
        Write-Host "Location: $($pythonPath.Source)" -ForegroundColor Gray
    } else {
        Write-Host "Python installed. Please restart your terminal to use it." -ForegroundColor Yellow
    }
}

if(-not (Test-Path "requirements.txt")) {
    Write-Host "requirements.txt not found";
    exit 0;
} else {
    Write-Host "Creating virtual enviorment...";
    python3 -m venv venv
    ./venv/Scripts/activate    
    Write-Host "Downloading libraries...";
    pip install -r requirements.txt
    deactivate
}

