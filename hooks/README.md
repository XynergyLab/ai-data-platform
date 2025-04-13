# Git Hooks for AI and Data Processing Platform

This directory contains Git hooks that help enforce development standards and security practices for the AI and Data Processing Platform.

## Available Hooks

### pre-commit

The `pre-commit` hook verifies that your commits are properly signed with a GPG key before allowing them to be committed. This hook:

- Checks if `commit.gpgsign` is enabled in your Git configuration
- Verifies you have a GPG signing key configured
- Ensures the configured key is valid and available in your keyring
- Provides helpful guidance if any of these checks fail

## Why GPG Signing?

GPG signing is required for this repository to:

1. **Verify Authenticity**: Ensure that commits are actually from the claimed author
2. **Prevent Tampering**: Protect against unauthorized modifications to the codebase
3. **Security Compliance**: Meet security standards for AI and data processing systems
4. **Non-Repudiation**: Provide cryptographic proof of who made specific changes
5. **Build Trust**: Establish a chain of trust within our development process

As we're dealing with AI models, data processing pipelines, and potentially sensitive information, it's essential that all code changes can be cryptographically verified to maintain the integrity and security of the platform.

## Installation

### Automatic Installation (Recommended)

Run the setup script from the repository root:

```powershell
# On Windows
.\setup-git-hooks.ps1

# On Linux/macOS
pwsh setup-git-hooks.ps1
```

This script will:
- Install the pre-commit hook
- Verify your GPG configuration
- Provide guidance if your GPG setup needs adjustment

### Manual Installation

If you prefer manual installation:

1. Copy the pre-commit hook to your local `.git/hooks` directory:
   ```bash
   cp hooks/pre-commit .git/hooks/pre-commit
   ```

2. Make it executable (on Unix-like systems):
   ```bash
   chmod +x .git/hooks/pre-commit
   ```

## Setting Up GPG Keys

If you haven't set up GPG signing for Git yet, you'll need to:

1. **Install GPG**:
   - Windows: [Gpg4win](https://www.gpg4win.org/)
   - macOS: `brew install gnupg`
   - Linux: `apt install gnupg` or `dnf install gnupg`

2. **Generate a GPG key** and configure Git to use it.

3. **Add your public key** to your GitHub account.

Detailed instructions are available in:
- [CONTRIBUTING.md → GPG Signing Requirements](../docs/CONTRIBUTING.md#gpg-signing-requirements)
- [GitHub Docs: Managing commit signature verification](https://docs.github.com/en/authentication/managing-commit-signature-verification)

## Troubleshooting

If you encounter issues with the pre-commit hook:

- Check that your GPG key is properly configured in Git
- Verify that your GPG key hasn't expired
- Ensure the GPG program path is correctly set in Git (especially on Windows)
- See the troubleshooting section in [CONTRIBUTING.md](../docs/CONTRIBUTING.md#troubleshooting-gpg-signing)

If you continue to experience problems, please reach out to the project maintainers.

