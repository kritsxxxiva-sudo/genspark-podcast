#!/usr/bin/env python3
"""
Test script for the Podcast AI Agents application
Tests basic functionality and API endpoints
"""

import subprocess
import time
import requests
import json
import os
import signal
from pathlib import Path

def test_demo_version():
    """Test the demo version of the podcast agents"""
    print("🧪 Testing Demo Version...")
    try:
        result = subprocess.run(
            ["python", "demo_podcast_agents.py"],
            cwd="/home/user/webapp",
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print("✅ Demo version works correctly!")
            return True
        else:
            print(f"❌ Demo version failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Demo version error: {e}")
        return False

def test_fastapi_app():
    """Test the FastAPI application"""
    print("🧪 Testing FastAPI Application...")
    
    # Start the server in background
    server_process = subprocess.Popen(
        ["python", "main.py"],
        cwd="/home/user/webapp",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working!")
            result = True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            result = False
    except Exception as e:
        print(f"❌ FastAPI test error: {e}")
        result = False
    finally:
        # Kill the server
        server_process.terminate()
        server_process.wait()
    
    return result

def test_file_generation():
    """Test if files are generated correctly"""
    print("🧪 Testing File Generation...")
    
    # Check if demo file was created
    demo_file = Path("/home/user/webapp/demo_podcast_episode.json")
    if demo_file.exists():
        try:
            with open(demo_file, 'r') as f:
                data = json.load(f)
                print(f"✅ Demo file created successfully!")
                print(f"📝 Episode title: {data.get('title', 'Unknown')}")
                return True
        except Exception as e:
            print(f"❌ Error reading demo file: {e}")
            return False
    else:
        print("❌ Demo file not found!")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Podcast AI Agents Tests...")
    print("=" * 50)
    
    # Test 1: Demo version
    demo_test = test_demo_version()
    
    # Test 2: File generation
    file_test = test_file_generation()
    
    # Test 3: FastAPI app
    api_test = test_fastapi_app()
    
    # Results
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"Demo Version: {'✅ PASS' if demo_test else '❌ FAIL'}")
    print(f"File Generation: {'✅ PASS' if file_test else '❌ FAIL'}")
    print(f"FastAPI App: {'✅ PASS' if api_test else '❌ FAIL'}")
    
    if all([demo_test, file_test, api_test]):
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())