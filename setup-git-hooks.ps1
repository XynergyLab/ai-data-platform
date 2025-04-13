<#
.SYNOPSIS
    Sets up Git hooks for the AI and Data Processing Platform repository.

.DESCRIPTION
    This script installs the pre-commit hook that enforces GPG signing of commits
    and verifies that GPG is properly configured in the user's environment.

.NOTES
    Author: AI and Data Processing Platform Team
    Date:   April 13, 2025
#>

# Output formatting functions
function Write-Header {
    param([string]$Message)
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host " $Message" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[✓] $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "[✗] ERROR: $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[!] WARNING: $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "[i] $Message" -ForegroundColor Blue
}

function Write-Step {
    param([string]$Message)
    Write-Host "`n>> $Message" -ForegroundColor Magenta
}

# Script starts here
Write-Header "AI and Data Processing Platform Git Hooks Setup"

# Step 1: Check if we're in a Git repository
Write-Step "Checking Git repository"
if (-not (Test-Path ".git")) {
    Write-Error "Not in a Git repository root directory."
    Write-Info "Please run this script from the root of the AI and Data Processing Platform repository."
    exit 1
}
Write-Success "Valid Git repository found"

# Step 2: Check if hooks directory exists, create if necessary
Write-Step "Setting up hooks directory"
$hooksDir = ".git/hooks"
if (-not (Test-Path $hooksDir)) {
    Write-Info "Hooks directory not found. Creating..."
    New-Item -ItemType Directory -Path $hooksDir -Force | Out-Null
    if (-not $?) {
        Write-Error "Failed to create hooks directory."
        exit 1
    }
    Write-Success "Hooks directory created"
} else {
    Write-Success "Hooks directory already exists"
}

# Step 3: Check if our pre-commit hook exists
Write-Step "Checking pre-commit hook source"
$sourceHook = "hooks/pre-commit"
if (-not (Test-Path $sourceHook)) {
    Write-Error "Source pre-commit hook not found at '$sourceHook'."
    Write-Info "Make sure you have the latest version of the repository."
    exit 1
}
Write-Success "Source pre-commit hook found"

# Step 4: Backup existing hook if necessary
$targetHook = ".git/hooks/pre-commit"
if (Test-Path $targetHook) {
    Write-Warning "Existing pre-commit hook found"
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupHook = "$targetHook.backup_$timestamp"
    Write-Info "Backing up existing hook to $backupHook"
    Copy-Item -Path $targetHook -Destination $backupHook -Force
    if (-not $?) {
        Write-Error "Failed to backup existing hook."
        exit 1
    }
    Write-Success "Existing hook backed up"
}

# Step 5: Copy the hook
Write-Step "Installing pre-commit hook"
Copy-Item -Path $sourceHook -Destination $targetHook -Force
if (-not $?) {
    Write-Error "Failed to copy pre-commit hook."
    exit 1
}
Write-Success "Pre-commit hook installed successfully"

# Step 6: Make hook executable (important for WSL/Git Bash)
if ($IsLinux -or $IsMacOS) {
    # For Unix-like systems
    & chmod +x $targetHook
    if (-not $?) {
        Write-Warning "Unable to make hook executable. You may need to run: chmod +x $targetHook"
    } else {
        Write-Success "Hook permissions set correctly"
    }
} else {
    # For Windows
    # Check if running in WSL/Git Bash environment where chmod might work
    try {
        $gitBashPath = (Get-Command -ErrorAction SilentlyContinue bash).Source
        if ($gitBashPath) {
            & bash -c "chmod +x '$($targetHook.Replace('\', '/'))'"
            Write-Success "Hook permissions set correctly via Git Bash"
        } else {
            Write-Info "Running on Windows - executable permissions not applicable, but hook should still work"
        }
    } catch {
        Write-Info "Running on Windows - executable permissions not applicable, but hook should still work"
    }
}

# Step 7: Verify GPG configuration
Write-Step "Verifying GPG configuration"

# Check if GPG is installed
$gpgInstalled = $false
try {
    $gpgVersion = & git config --get gpg.program
    if ([string]::IsNullOrEmpty($gpgVersion)) {
        try {
            $gpgTest = & gpg --version
            $gpgInstalled = $true
        } catch {
            $gpgInstalled = $false
        }
    } else {
        $gpgInstalled = $true
    }
} catch {
    $gpgInstalled = $false
}

if (-not $gpgInstalled) {
    Write-Warning "GPG does not appear to be installed or configured in Git"
    Write-Info "For Windows, we recommend installing Gpg4win: https://www.gpg4win.org/"
    Write-Info "For macOS: brew install gnupg"
    Write-Info "For Linux: apt install gnupg or dnf install gnupg"
} else {
    Write-Success "GPG is installed"
}

# Check if commit.gpgsign is enabled
$gpgSign = & git config --get commit.gpgsign
if ([string]::IsNullOrEmpty($gpgSign) -or $gpgSign -ne "true") {
    Write-Warning "GPG commit signing is not enabled"
    Write-Info "To enable automatic commit signing, run: git config --global commit.gpgsign true"
} else {
    Write-Success "GPG commit signing is enabled"
}

# Check if a signing key is configured
$signingKey = & git config --get user.signingkey
if ([string]::IsNullOrEmpty($signingKey)) {
    Write-Warning "No GPG signing key configured"
    Write-Info "To find your key ID, run: gpg --list-secret-keys --keyid-format=long"
    Write-Info "Then set it with: git config --global user.signingkey YOUR_KEY_ID"
} else {
    Write-Success "GPG signing key is configured: $signingKey"
    
    # Verify key exists in keyring
    try {
        $keyCheck = & gpg --list-keys $signingKey 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "GPG key found in keyring"
        } else {
            Write-Warning "The configured GPG key does not appear to be in your keyring"
            Write-Info "Make sure you've imported your key properly"
        }
    } catch {
        Write-Warning "Could not verify GPG key in keyring"
    }
}

# Check GPG program path on Windows
if ($IsWindows -or $env:OS -match "Windows") {
    $gpgProgram = & git config --get gpg.program
    if ([string]::IsNullOrEmpty($gpgProgram)) {
        Write-Warning "gpg.program path is not set, which may cause issues on Windows"
        Write-Info "To set the GPG program path, run: git config --global gpg.program `"C:/Program Files (x86)/GnuPG/bin/gpg.exe`""
        Write-Info "(Adjust the path to match your GPG installation)"
    } elseif (-not (Test-Path $gpgProgram)) {
        Write-Warning "The configured GPG program does not exist at: $gpgProgram"
        Write-Info "Please check the path and update it if necessary"
    } else {
        Write-Success "GPG program path is correctly configured"
    }
}

# Final instructions
Write-Header "Setup Complete"
Write-Info "The GPG verification pre-commit hook has been installed."
Write-Info "This hook will verify that your commits are properly signed with your GPG key."
Write-Info "For more information on GPG signing requirements, see:"
Write-Info "  docs/CONTRIBUTING.md#gpg-signing-requirements"
Write-Info "If you experience any issues, please contact the project maintainers."

