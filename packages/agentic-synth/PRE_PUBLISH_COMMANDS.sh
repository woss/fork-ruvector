#!/bin/bash
# Agentic-Synth Pre-Publish Fix Commands
# Run these commands to fix all critical issues before npm publish

set -e  # Exit on any error

echo "🚀 Agentic-Synth Pre-Publish Fix Script"
echo "========================================"
echo ""

# 1. Enable TypeScript declarations
echo "✓ Step 1: Enabling TypeScript declarations..."
sed -i 's/"declaration": false/"declaration": true/' tsconfig.json
echo "  Declaration enabled in tsconfig.json"
echo ""

# 2. Fix variable shadowing bug
echo "✓ Step 2: Fixing variable shadowing bug..."
sed -i '548s/const performance =/const performanceMetrics =/' training/dspy-learning-session.ts
echo "  Fixed performance variable shadowing"
echo ""

# 3. Rebuild with type definitions
echo "✓ Step 3: Rebuilding package with type definitions..."
npm run build:all
echo "  Build complete!"
echo ""

# 4. Verify type definitions created
echo "✓ Step 4: Verifying .d.ts files..."
if [ -f "dist/index.d.ts" ] && [ -f "dist/cache/index.d.ts" ] && [ -f "dist/generators/index.d.ts" ]; then
    echo "  ✅ All type definition files created successfully!"
else
    echo "  ❌ ERROR: Some .d.ts files missing!"
    find dist -name "*.d.ts"
    exit 1
fi
echo ""

# 5. Test npm pack
echo "✓ Step 5: Testing npm pack..."
npm pack --dry-run | head -20
echo ""

# 6. Show next steps
echo "========================================"
echo "✅ All automated fixes complete!"
echo ""
echo "📝 Manual Steps Required:"
echo ""
echo "1. Edit package.json:"
echo "   - Move 'types' field BEFORE 'import' in all 3 exports"
echo "   - Update 'files' field to: [\"dist\", \"bin\", \"config\", \"README.md\", \"LICENSE\"]"
echo ""
echo "2. Test local installation:"
echo "   npm pack"
echo "   npm install -g ./ruvector-agentic-synth-0.1.0.tgz"
echo "   agentic-synth --version"
echo "   agentic-synth validate"
echo ""
echo "3. Verify TypeScript imports work:"
echo "   Create test.ts and import package"
echo "   npx tsc --noEmit test.ts"
echo ""
echo "4. Publish to npm:"
echo "   npm publish --access public --dry-run  # Test first"
echo "   npm publish --access public            # Real publish"
echo ""
echo "🚀 Ready for publication after manual steps!"
