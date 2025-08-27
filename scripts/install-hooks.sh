#!/bin/bash
#
# Install Git hooks for the OCR Research Project
# This script installs pre-push hooks to automatically test links

set -e

# Change to project root
cd "$(git rev-parse --show-toplevel)"

echo "🔧 Installing Git hooks for OCR Research Project..."

# Create hooks directory if it doesn't exist
mkdir -p .git/hooks

# Install pre-push hook
if [ -f "scripts/pre-push-hook.sh" ]; then
    cp scripts/pre-push-hook.sh .git/hooks/pre-push
    chmod +x .git/hooks/pre-push
    echo "✅ Pre-push hook installed successfully!"
    echo "   This will automatically test all links before pushing to GitHub"
else
    echo "❌ Pre-push hook script not found: scripts/pre-push-hook.sh"
    exit 1
fi

echo ""
echo "🎉 Git hooks installation complete!"
echo ""
echo "The following hooks are now active:"
echo "  • pre-push: Tests all GitHub Pages links before push"
echo ""
echo "To disable hooks temporarily, use: git push --no-verify"

exit 0