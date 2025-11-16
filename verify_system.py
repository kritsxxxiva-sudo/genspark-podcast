#!/usr/bin/env python3
"""
Final verification script for the Podcast AI Agents application
Comprehensive testing of all components
"""

import subprocess
import json
import os
from pathlib import Path

def verify_installation():
    """Verify all components are properly installed"""
    print("🔍 Verifying Installation...")
    
    # Check Python version
    result = subprocess.run(["python", "--version"], capture_output=True, text=True)
    print(f"Python Version: {result.stdout.strip()}")
    
    # Check required packages
    packages = ["fastapi", "uvicorn", "openai", "pydub", "python-dotenv"]
    for package in packages:
        result = subprocess.run(["pip", "show", package], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {package} installed")
        else:
            print(f"❌ {package} missing")
            return False
    
    return True

def verify_files():
    """Verify all required files exist"""
    print("\n📁 Verifying Files...")
    
    required_files = [
        "main.py",
        "demo_podcast_agents.py", 
        "enhanced_podcast_agents.py",
        "advanced_podcast_agents.py",
        "config.py",
        "requirements.txt",
        ".env",
        "test_system.py",
        "demo_podcast_episode.json"
    ]
    
    for file_path in required_files:
        if Path(f"/home/user/webapp/{file_path}").exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    return True

def verify_directories():
    """Verify required directories exist"""
    print("\n📂 Verifying Directories...")
    
    dirs = ["episodes", "audio", "logs", "temp"]
    for dir_name in dirs:
        if Path(f"/home/user/webapp/{dir_name}").exists():
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
            return False
    
    return True

def verify_demo_functionality():
    """Verify demo version works"""
    print("\n🎭 Verifying Demo Functionality...")
    
    try:
        result = subprocess.run(
            ["python", "demo_podcast_agents.py"],
            cwd="/home/user/webapp",
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and "Demo completed!" in result.stdout:
            print("✅ Demo version functional")
            return True
        else:
            print(f"❌ Demo version failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Demo test error: {e}")
        return False

def verify_api_functionality():
    """Verify API endpoints work"""
    print("\n🌐 Verifying API Functionality...")
    
    try:
        # Test FastAPI import
        result = subprocess.run(
            ["python", "-c", "from main import app; print('FastAPI import successful')"],
            cwd="/home/user/webapp",
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ FastAPI app can be imported")
            return True
        else:
            print(f"❌ FastAPI import failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ API test error: {e}")
        return False

def verify_git_status():
    """Verify git repository status"""
    print("\n📊 Verifying Git Repository...")
    
    try:
        result = subprocess.run(
            ["git", "status"],
            cwd="/home/user/webapp",
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Git repository accessible")
            
            # Check remote
            remote_result = subprocess.run(
                ["git", "remote", "-v"],
                cwd="/home/user/webapp",
                capture_output=True,
                text=True
            )
            
            if "github.com" in remote_result.stdout:
                print("✅ GitHub repository connected")
                return True
            else:
                print("⚠️  GitHub repository not detected")
                return True  # Still functional
        else:
            print(f"❌ Git repository error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Git test error: {e}")
        return False

def main():
    """Run all verification tests"""
    print("🚀 Final Verification of Podcast AI Agents System")
    print("=" * 60)
    
    tests = [
        ("Installation", verify_installation),
        ("Files", verify_files),
        ("Directories", verify_directories),
        ("Demo Functionality", verify_demo_functionality),
        ("API Functionality", verify_api_functionality),
        ("Git Repository", verify_git_status)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\n📊 Overall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 System fully verified and ready for use!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())