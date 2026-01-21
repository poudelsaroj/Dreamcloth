"""
Syntax checker for Phase 3 module
Verifies all Python files have valid syntax without requiring dependencies
"""

import py_compile
import os
import sys

def check_syntax(filepath):
    """Check if a Python file has valid syntax"""
    try:
        py_compile.compile(filepath, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)

def main():
    """Check syntax of all Python files in phase3"""
    files = [
        'model.py',
        'utils.py',
        'dataloader.py',
        'train.py',
        'test.py',
        '__init__.py'
    ]

    print("="*60)
    print("Phase 3 Syntax Verification")
    print("="*60)

    all_ok = True
    for filename in files:
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if not os.path.exists(filepath):
            print(f"[ERROR] {filename}: FILE NOT FOUND")
            all_ok = False
            continue

        ok, error = check_syntax(filepath)
        if ok:
            # Count lines
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
            print(f"[OK] {filename}: Valid syntax ({lines} lines)")
        else:
            print(f"[ERROR] {filename}: SYNTAX ERROR")
            print(f"  {error}")
            all_ok = False

    print("="*60)
    if all_ok:
        print("[SUCCESS] All files have valid syntax!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run tests: python test.py")
        print("3. Start training: python train.py")
    else:
        print("[FAILED] Some files have syntax errors")
        sys.exit(1)

if __name__ == "__main__":
    main()
