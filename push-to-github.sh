#!/bin/bash

# GitHub Push Helper Script
# Health MLOps Project

echo "════════════════════════════════════════════════════════"
echo "🚀 GITHUB PUSH HELPER - Health MLOps Project"
echo "════════════════════════════════════════════════════════"
echo ""

# Check if we're in a git repo
if [ ! -d .git ]; then
    echo "❌ Error: Not a git repository!"
    echo "   Run this script from the project root directory"
    exit 1
fi

# Check git status
echo "📊 Current Git Status:"
echo "---"
git status --short
echo ""

# Check commits
COMMIT_COUNT=$(git rev-list --count HEAD)
echo "📝 Commits ready: $COMMIT_COUNT"
echo ""

# Check if remote exists
if git remote get-url origin > /dev/null 2>&1; then
    CURRENT_REMOTE=$(git remote get-url origin)
    echo "🔗 Current remote: $CURRENT_REMOTE"
    echo ""
    echo "⚠️  Remote already configured!"
    echo ""
    read -p "Do you want to update it? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your GitHub username: " USERNAME
        NEW_REMOTE="https://github.com/$USERNAME/health-mlops-project.git"
        echo "Updating remote to: $NEW_REMOTE"
        git remote set-url origin "$NEW_REMOTE"
        echo "✅ Remote updated!"
    fi
else
    echo "🔗 No remote configured yet"
    echo ""
    read -p "Enter your GitHub username: " USERNAME

    if [ -z "$USERNAME" ]; then
        echo "❌ Username cannot be empty!"
        exit 1
    fi

    REMOTE_URL="https://github.com/$USERNAME/health-mlops-project.git"
    echo ""
    echo "Adding remote: $REMOTE_URL"
    git remote add origin "$REMOTE_URL"
    echo "✅ Remote added!"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "🎯 READY TO PUSH"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Before pushing, make sure you have:"
echo "  1. ✅ Created repository on GitHub"
echo "  2. ✅ Named it: health-mlops-project"
echo "  3. ✅ Made it Public"
echo "  4. ✅ Did NOT initialize with README"
echo ""
read -p "Have you done all the above? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "⚠️  Please create the repository first:"
    echo "   1. Go to https://github.com/new"
    echo "   2. Create 'health-mlops-project'"
    echo "   3. Then run this script again"
    exit 1
fi

echo ""
echo "🚀 Pushing to GitHub..."
echo ""

# Push to GitHub
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "✅ SUCCESS! Your project is now on GitHub!"
    echo "════════════════════════════════════════════════════════"
    echo ""
    REMOTE_URL=$(git remote get-url origin)
    REPO_URL=${REMOTE_URL%.git}
    echo "🌐 View your repository:"
    echo "   $REPO_URL"
    echo ""
    echo "🎬 View CI/CD Actions:"
    echo "   $REPO_URL/actions"
    echo ""
    echo "📊 Your CI/CD pipeline will start automatically!"
    echo ""
    echo "Next steps:"
    echo "  1. Open your repo in browser"
    echo "  2. Click 'Actions' tab to see CI/CD running"
    echo "  3. Add topics/tags (mlops, federated-learning, etc.)"
    echo "  4. Share link with professor"
    echo ""
    echo "🎉 You're all set for your presentation!"
else
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "❌ PUSH FAILED"
    echo "════════════════════════════════════════════════════════"
    echo ""
    echo "Common issues:"
    echo "  1. Repository doesn't exist on GitHub"
    echo "  2. Authentication failed (use Personal Access Token)"
    echo "  3. Branch protection rules"
    echo ""
    echo "📖 Check GITHUB_SETUP.md for detailed instructions"
    exit 1
fi
