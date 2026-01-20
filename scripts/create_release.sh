#!/bin/bash
# Create GitHub release for OpenBoost 1.0.0rc1
# Run this after pushing the tag: git tag v1.0.0rc1 && git push origin v1.0.0rc1

set -e

VERSION="1.0.0rc1"
TAG="v${VERSION}"

echo "Creating GitHub release for ${TAG}..."

# Check if tag exists
if ! git rev-parse "${TAG}" >/dev/null 2>&1; then
    echo "Error: Tag ${TAG} does not exist."
    echo "Run: git tag ${TAG} && git push origin ${TAG}"
    exit 1
fi

# Create release with notes
gh release create "${TAG}" \
    --title "OpenBoost ${VERSION}" \
    --notes-file docs/releases/RELEASE_NOTES_${VERSION}.md \
    --prerelease

echo ""
echo "✅ Release created!"
echo ""
echo "Next steps:"
echo "1. The PyPI publish workflow will run automatically"
echo "2. Post announcements from docs/releases/ANNOUNCEMENTS.md:"
echo "   - Twitter/X thread"
echo "   - LinkedIn post"  
echo "   - Reddit r/MachineLearning"
echo "   - Hacker News Show HN"
echo ""
echo "Release URL: https://github.com/jxucoder/openboost/releases/tag/${TAG}"
