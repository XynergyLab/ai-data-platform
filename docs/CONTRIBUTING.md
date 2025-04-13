# Contributing to the AI and Data Processing Platform

First off, thank you for considering contributing to the AI and Data Processing Platform! It's people like you that make this platform a powerful tool for AI services, data processing, and monitoring.

This document provides guidelines and instructions for contributing to the project. By participating in this project, you agree to abide by our [Code of Conduct](../.github/CODE_OF_CONDUCT.md).

## Table of Contents

- [Getting Started](#getting-started)
  - [Setting Up Development Environment](#setting-up-development-environment)
  - [Project Structure](#project-structure)
- [Contribution Workflow](#contribution-workflow)
  - [Fork and Clone](#fork-and-clone)
  - [Branching Strategy](#branching-strategy)
  - [Commit Guidelines](#commit-guidelines)
  - [Pull Requests](#pull-requests)
- [Coding Standards](#coding-standards)
  - [Python Code](#python-code)
  - [YAML Configuration](#yaml-configuration)
  - [PowerShell Scripts](#powershell-scripts)
  - [Documentation](#documentation)
- [Testing Requirements](#testing-requirements)
  - [Unit Testing](#unit-testing)
  - [Integration Testing](#integration-testing)
  - [Manual Testing](#manual-testing)
- [Documentation Guidelines](#documentation-guidelines)
- [Pull Request Review Process](#pull-request-review-process)
- [Community](#community)

## Getting Started

### Setting Up Development Environment

#### Prerequisites

- Git
- Podman (version 3.0+)
- Python 3.10+
- PowerShell 7+ (for Windows users)
- WSL2 (for Windows users)

#### Installation Steps

1. **Install Podman**

   For Windows (using WSL2):
   ```powershell
   # Ensure WSL2 is installed and configured
   wsl --set-default-version 2
   
   # Install Podman on Windows
   winget install RedHat.Podman
   ```

   For Linux:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get -y install podman
   
   # Fedora
   sudo dnf -y install podman
   ```

2. **Clone the repository**

   ```bash
   git clone https://github.com/[YourUsername]/ai-data-platform.git
   cd ai-data-platform
   ```

3. **Set up Python environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Verify Podman setup**

   ```bash
   podman --version
   podman info
   ```

### Project Structure

The project is organized into several key directories:

- `ai-services/` - LLM inference, embedding, and multimodal services
- `data-processing/` - Data pipelines and batch processing jobs
- `databases/` - Vector stores and database configurations
- `monitoring/` - Metrics, logging, and dashboards
- `ml-infrastructure/` - ML model serving and training infrastructure
- `docs/` - Documentation
- `.github/` - GitHub workflows and templates

## Contribution Workflow

### Fork and Clone

1. Fork the [main repository](https://github.com/XynergyLab/ai-data-platform)
2. Clone your fork locally
3. Add the original repository as a remote named "upstream"
   ```bash
   git remote add upstream https://github.com/XynergyLab/ai-data-platform.git
   ```

### Branching Strategy

We follow a specific branch naming convention:

- `feature/[issue-id]-short-description` - For new features
- `bugfix/[issue-id]-short-description` - For bug fixes
- `docs/[issue-id]-short-description` - For documentation changes
- `release/v[major].[minor].[patch]` - For release preparation

Always create branches from `develop`, not `main`.

```bash
git checkout develop
git pull upstream develop
git checkout -b feature/123-new-embedding-model
```

### Commit Guidelines

Follow these best practices for commits:

- Write clear, concise commit messages
- Use the imperative mood ("Add feature" not "Added feature")
- Reference issues in commit messages (e.g., "Fix #123: Resolve memory leak in embedding service")
- Keep commits focused on a single change
- Squash multiple commits when they address the same logical change

Example commit message:
```
Add Milvus vector store integration

- Add Milvus Docker configuration
- Implement Python client for vector operations
- Add health check endpoint
- Update documentation with setup instructions

Fixes #123
```

### Pull Requests

1. Push your branch to your fork
   ```bash
   git push origin feature/123-new-embedding-model
   ```

2. Open a pull request against the `develop` branch
3. Fill out the PR template completely
4. Request reviews from appropriate team members
5. Respond to any feedback and make necessary changes
6. Once approved, your PR will be merged

## Coding Standards

### Python Code

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use type hints for function parameters and return values
- Document classes and functions using docstrings
- Maximum line length of 100 characters
- Use meaningful variable and function names
- Write unit tests for new functionality

Example:
```python
def process_embedding(
    text: str,
    model_name: str = "default-embedding-model"
) -> np.ndarray:
    """
    Generate vector embeddings for the provided text.
    
    Args:
        text: The input text to embed
        model_name: The embedding model to use
        
    Returns:
        A numpy array representing the text embedding
        
    Raises:
        ModelNotFoundError: If the specified model cannot be loaded
    """
    # Implementation
```

### YAML Configuration

- Use 2-space indentation
- Use snake_case for keys
- Include comments for non-obvious configuration options
- Keep related settings grouped together
- Validate all YAML files before committing

### PowerShell Scripts

- Follow the [PowerShell Best Practices](https://docs.microsoft.com/en-us/powershell/scripting/developer/cmdlet/cmdlet-development-guidelines)
- Use Pascal case for function names
- Include comment-based help for all functions
- Use parameter validation attributes
- Handle errors appropriately

### Documentation

- Write in Markdown
- Use proper headings and hierarchy
- Include code examples where appropriate
- Keep documentation up-to-date with code changes
- Add diagrams for complex systems

## Testing Requirements

### Unit Testing

All new code should include unit tests:

- Use pytest for Python code
- Aim for at least 80% code coverage
- Mock external dependencies
- Keep tests fast and isolated

Running unit tests:
```bash
cd ai-services
pytest -xvs tests/
```

### Integration Testing

- Write integration tests for service interactions
- Test Podman deployments using docker-compose
- Validate API endpoints and data flows

Running integration tests:
```bash
cd tests/integration
python -m pytest
```

### Manual Testing

Some features require manual testing:

1. Spin up the relevant services with Podman
   ```bash
   cd ai-services/llm-inference
   podman-compose up -d
   ```

2. Test the functionality via the relevant interface (API, UI, etc.)
3. Document your test results in the PR

## Documentation Guidelines

Good documentation is critical to the project's success:

- Update README files for components you modify
- Add inline code comments for complex logic
- Create or update architecture diagrams when changing system design
- Document API endpoints using OpenAPI/Swagger
- Include examples for new features
- Add environment variables to the relevant .env.example files

## Pull Request Review Process

1. Automated checks must pass (CI/CD workflows)
2. At least two maintainer approvals are required
3. All PR comments must be resolved
4. Documentation must be updated
5. Testing requirements must be met
6. Code must follow the established standards

Reviewers will check for:
- Code correctness and quality
- Test coverage
- Documentation completeness
- Performance considerations
- Security implications

## Community

Join our community channels:

- GitHub Discussions: For feature ideas and general questions
- Issue Tracker: For bugs and specific issues
- Discord: For real-time discussion and collaboration

We hold bi-weekly contributor meetings - check the GitHub Discussions for schedule.

---

Thank you for contributing to the AI and Data Processing Platform! Your efforts help make this project better for everyone.

