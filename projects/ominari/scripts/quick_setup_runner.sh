#!/bin/bash

# Quick GitHub Actions Runner Setup Guide

echo "🚀 Quick Setup for GitHub Actions Self-Hosted Runner"
echo "=================================================="
echo
echo "This will set up automatic deployment when you push to main."
echo

# Get repo info
REPO_URL=$(git remote get-url origin)
echo "Repository: $REPO_URL"
echo

echo "📋 STEP 1: Get your runner token"
echo "--------------------------------"
echo "1. Open this URL in your browser:"
echo "   $REPO_URL/settings/actions/runners/new"
echo
echo "2. You'll see a registration token that looks like: AAAA..."
echo "3. Copy that token"
echo

read -p "Paste your runner token here: " TOKEN
echo

echo "📦 STEP 2: Running automated setup..."
echo "------------------------------------"

# Run the main setup script
./scripts/setup_github_runner.sh

echo
echo "✅ Setup complete!"
echo
echo "🔄 What happens now:"
echo "- Every time you push to main, Ominari will automatically:"
echo "  1. Stop the old version"
echo "  2. Pull the new code"  
echo "  3. Install any new dependencies"
echo "  4. Start the trading system"
echo "  5. Open the dashboard"
echo
echo "🎯 Try it out:"
echo "1. Make a small change (like updating README.md)"
echo "2. Commit and push to main:"
echo "   git add ."
echo "   git commit -m 'Test auto-deployment'"
echo "   git push origin main"
echo
echo "3. Watch the magic happen at:"
echo "   $REPO_URL/actions"
echo
echo "The system will keep running even after you log out!"