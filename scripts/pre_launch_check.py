"""Pre-launch verification script."""

import subprocess
import sys


def check_version():
    """Verify version consistency."""
    print("Checking version...")
    import openboost
    
    # Read pyproject.toml version
    try:
        import tomllib
        with open("pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
    except ImportError:
        # Python < 3.11 fallback
        import re
        with open("pyproject.toml") as f:
            content = f.read()
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        if match:
            pyproject = {"project": {"version": match.group(1)}}
        else:
            print("❌ Could not parse pyproject.toml version")
            return False
    
    pyproject_version = pyproject["project"]["version"]
    
    if openboost.__version__ != pyproject_version:
        print(f"❌ Version mismatch: __init__.py={openboost.__version__}, pyproject.toml={pyproject_version}")
        return False
    
    print(f"✅ Version: {openboost.__version__}")
    return True


def check_imports():
    """Verify all public imports work."""
    print("\nChecking imports...")
    import openboost as ob
    
    required = [
        "GradientBoosting",
        "NaturalBoostNormal",
        "OpenBoostGAM",
        "DART",
        "OpenBoostRegressor",
        "EarlyStopping",
        "compute_feature_importances",
    ]
    
    missing = []
    for name in required:
        if not hasattr(ob, name):
            missing.append(name)
    
    if missing:
        print(f"❌ Missing: {missing}")
        return False
    
    print(f"✅ All {len(required)} required imports available")
    return True


def run_tests():
    """Run test suite."""
    print("\nRunning tests...")
    result = subprocess.run(
        ["uv", "run", "pytest", "tests/", "-v", "--tb=short", "-q"],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"❌ Tests failed")
        print(result.stdout[-2000:])  # Last 2000 chars
        return False
    
    print("✅ All tests pass")
    return True


def check_readme_examples():
    """Verify README examples work."""
    print("\nChecking README examples...")
    result = subprocess.run(
        ["uv", "run", "pytest", "tests/test_readme_examples.py", "-v"],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"❌ README examples failed")
        return False
    
    print("✅ README examples work")
    return True


def check_package_build():
    """Verify package builds."""
    print("\nBuilding package...")
    result = subprocess.run(
        ["uv", "build"],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"❌ Package build failed")
        return False
    
    print("✅ Package builds successfully")
    return True


def main():
    print("=" * 60)
    print("OPENBOOST PRE-LAUNCH CHECK")
    print("=" * 60)
    
    checks = [
        ("Version", check_version),
        ("Imports", check_imports),
        ("Tests", run_tests),
        # ("README Examples", check_readme_examples),  # Enable when created
        ("Package Build", check_package_build),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results[name] = False
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_pass = True
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_pass = False
    
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ READY FOR LAUNCH!")
    else:
        print("❌ FIX ISSUES BEFORE LAUNCH")
    print("=" * 60)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
