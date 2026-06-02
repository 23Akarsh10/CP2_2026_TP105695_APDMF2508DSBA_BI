param(
    [string]$Message = "Initial commit"
)

function Check-Git {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if (-not $git) {
        Write-Host "Git is not installed or not found in PATH. Please install Git: https://git-scm.com/download/win" -ForegroundColor Yellow
        exit 1
    }
}

Check-Git

git init
git add .
git commit -m $Message
Write-Host "Repository initialized and committed." -ForegroundColor Green
