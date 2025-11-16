#!/usr/bin/env python3
"""
Podcast AI Agents - Setup and Installation Script
Run this script to set up the application environment
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(cmd, description=""):
    """Run a shell command and handle errors."""
    print(f"Running: {cmd}")
    if description:
        print(f"Description: {description}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {cmd}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {cmd}")
        print(f"Error: {e.stderr}")
        return None

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python version: {sys.version}")

def install_dependencies():
    """Install required Python packages."""
    print("📦 Installing dependencies...")
    
    # Upgrade pip
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if Path("requirements.txt").exists():
        run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements")
    else:
        print("⚠️  requirements.txt not found, skipping dependency installation")

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    directories = [
        "episodes",
        "audio",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def setup_environment():
    """Set up environment configuration."""
    print("🔧 Setting up environment...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        # Copy .env.example to .env
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("✅ Created .env file from .env.example")
        print("⚠️  Please edit .env file and add your API keys")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️  No .env.example file found, please create .env manually")

def create_sample_files():
    """Create sample files for testing."""
    print("📝 Creating sample files...")
    
    # Create sample topics file
    topics_file = Path("sample_topics.json")
    if not topics_file.exists():
        sample_topics = {
            "topics": [
                "The Future of AI Agents",
                "Machine Learning in Healthcare",
                "Climate Change and Technology",
                "Space Exploration Innovations",
                "Quantum Computing Breakthroughs",
                "Blockchain Beyond Cryptocurrency",
                "The Ethics of Artificial Intelligence",
                "Cybersecurity in the Digital Age",
                "Renewable Energy Technologies",
                "Biotechnology and Human Enhancement"
            ]
        }
        
        with open(topics_file, 'w') as f:
            json.dump(sample_topics, f, indent=2)
        print(f"✅ Created sample topics file: {topics_file}")

def main():
    """Main setup function."""
    print("🚀 Podcast AI Agents - Setup Script")
    print("=" * 50)
    
    try:
        # Check Python version
        check_python_version()
        
        # Install dependencies
        install_dependencies()
        
        # Create directories
        create_directories()
        
        # Setup environment
        setup_environment()
        
        # Create sample files
        create_sample_files()
        
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file and add your API keys")
        print("2. Run: python main.py")
        print("3. Visit: http://localhost:8000/docs for API documentation")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()