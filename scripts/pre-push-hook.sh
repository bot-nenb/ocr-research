#!/bin/bash
#
# Pre-push hook script for OCR Research Project
# Automatically runs link tests before pushing to GitHub
#
# To install this hook, copy it to .git/hooks/pre-push and make executable:
#   cp scripts/pre-push-hook.sh .git/hooks/pre-push
#   chmod +x .git/hooks/pre-push

set -e

echo "🔍 Running pre-push checks for OCR Research Project..."
echo "======================================================"

# Change to project root directory
cd "$(git rev-parse --show-toplevel)"

# Check if docs directory exists
if [ ! -d "docs" ]; then
    echo "⚠️  No docs directory found, skipping link tests"
    exit 0
fi

# Run link tests
echo "🧪 Testing all links in GitHub Pages..."
if python3 scripts/test_links.py; then
    echo "✅ All link tests passed!"
else
    echo "❌ Link tests failed!"
    echo ""
    echo "🔧 Please fix the broken links before pushing:"
    echo "   1. Check the link_test_report.json for details"
    echo "   2. Fix broken links in HTML files"
    echo "   3. Run 'python3 scripts/test_links.py' to verify fixes"
    echo "   4. Commit your fixes and try pushing again"
    echo ""
    exit 1
fi

# Optional: Run additional checks here
# - Check for large files
# - Validate JSON files
# - Check image formats
# etc.

echo ""
echo "🎉 All pre-push checks passed! Proceeding with push..."
echo "======================================================"

exit 0