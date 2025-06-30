#!/usr/bin/env python3
"""
Setup script for LM Training Control Panel
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(command, description):
    """Run a shell command with error handling."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return None


def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    dependencies = {
        "python": "python --version",
        "git": "git --version",
        "pip": "pip --version"
    }
    
    missing = []
    for dep, cmd in dependencies.items():
        if not shutil.which(dep.split()[0]):
            missing.append(dep)
        else:
            result = run_command(cmd, f"Checking {dep}")
            if result:
                print(f"   {dep}: {result.strip()}")
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Please install the missing dependencies and run setup again.")
        sys.exit(1)
    
    print("✅ All dependencies are available")


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "data",
        "data/raw",
        "data/processed",
        "data/baselines",
        "models",
        "experiments",
        "logs",
        "temp",
        "config/grafana/dashboards",
        "config/grafana/datasources"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Directories created successfully")


def setup_environment():
    """Set up Python environment and install dependencies."""
    print("🐍 Setting up Python environment...")
    
    # Check if virtual environment exists
    if not Path("venv").exists():
        run_command("python -m venv venv", "Creating virtual environment")
    
    # Determine activation script based on OS
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    # Install dependencies
    run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")
    run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies")
    
    print("✅ Python environment setup completed")
    return python_cmd, pip_cmd


def setup_configuration():
    """Set up configuration files."""
    print("⚙️  Setting up configuration...")
    
    # Copy environment file if it doesn't exist
    if not Path(".env").exists():
        if Path(".env.example").exists():
            shutil.copy(".env.example", ".env")
            print("   Created .env from .env.example")
            print("   ⚠️  Please edit .env file with your configuration")
        else:
            print("   ❌ .env.example not found")
    else:
        print("   .env already exists")
    
    print("✅ Configuration setup completed")


def initialize_dvc():
    """Initialize DVC for data versioning."""
    print("📊 Initializing DVC...")
    
    if not Path(".dvc").exists():
        result = run_command("dvc init", "Initializing DVC")
        if result is not None:
            run_command("git add .dvc/", "Adding DVC to git")
            print("   DVC initialized successfully")
    else:
        print("   DVC already initialized")
    
    print("✅ DVC setup completed")


def setup_git_hooks():
    """Set up pre-commit hooks."""
    print("🪝 Setting up git hooks...")
    
    # Install pre-commit if not already installed
    pip_cmd = "venv\\Scripts\\pip" if os.name == 'nt' else "venv/bin/pip"
    run_command(f"{pip_cmd} install pre-commit", "Installing pre-commit")
    
    # Install hooks
    if os.name == 'nt':
        run_command("venv\\Scripts\\pre-commit install", "Installing pre-commit hooks")
    else:
        run_command("venv/bin/pre-commit install", "Installing pre-commit hooks")
    
    print("✅ Git hooks setup completed")


def run_initial_tests():
    """Run initial tests to verify setup."""
    print("🧪 Running initial tests...")
    
    python_cmd = "venv\\Scripts\\python" if os.name == 'nt' else "venv/bin/python"
    
    # Test import of main modules
    test_imports = [
        "src.core.config",
        "src.main"
    ]
    
    for module in test_imports:
        result = run_command(f"{python_cmd} -c 'import {module}; print(\"✅ {module}\")'", 
                           f"Testing import of {module}")
        if result is None:
            print(f"   ❌ Failed to import {module}")
    
    print("✅ Initial tests completed")


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Edit .env file with your configuration:")
    print("   - Database connection string")
    print("   - API keys (if needed)")
    print("   - Storage configuration")
    print()
    print("2. Start the application:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("   python -m src.main")
    print()
    print("3. Or use Docker:")
    print("   docker-compose up -d")
    print()
    print("4. Access the application:")
    print("   - API: http://localhost:8000")
    print("   - API Docs: http://localhost:8000/docs")
    print("   - MLflow: http://localhost:5000")
    print("   - Grafana: http://localhost:3000 (admin/admin)")
    print()
    print("5. For development:")
    print("   make dev-install  # Install development dependencies")
    print("   make test         # Run tests")
    print("   make lint         # Run linting")
    print("   make format       # Format code")
    print()
    print("📚 See docs/README.md for detailed documentation")


def main():
    """Main setup function."""
    print("🚀 LM Training Control Panel Setup")
    print("="*60)
    
    try:
        check_dependencies()
        create_directories()
        python_cmd, pip_cmd = setup_environment()
        setup_configuration()
        initialize_dvc()
        setup_git_hooks()
        run_initial_tests()
        print_next_steps()
        
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
