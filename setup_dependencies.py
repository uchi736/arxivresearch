#!/usr/bin/env python3
"""
Setup script to ensure all dependencies are installed
"""
import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_dependencies():
    """Check and install missing dependencies"""
    required_packages = [
        "openpyxl>=3.0.0",  # For Excel export
        "pandas>=1.3.0",    # For CSV processing
        "beautifulsoup4",   # For HTML processing
        "requests",         # For web requests
    ]
    
    print("Checking dependencies...")
    
    for package in required_packages:
        package_name = package.split(">=")[0]
        try:
            __import__(package_name)
            print(f"✓ {package_name} is installed")
        except ImportError:
            print(f"✗ {package_name} is missing, installing...")
            if install_package(package):
                print(f"✓ {package_name} installed successfully")
            else:
                print(f"✗ Failed to install {package_name}")
                return False
    
    print("\nAll dependencies are ready!")
    return True

if __name__ == "__main__":
    if check_dependencies():
        print("\n[SUCCESS] System is ready to use!")
        print("You can now run:")
        print("  python cli_app.py registry export --output data.xlsx")
    else:
        print("\n[ERROR] Some dependencies failed to install")
        sys.exit(1)