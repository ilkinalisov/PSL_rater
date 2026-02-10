#!/bin/bash
# quick_setup.sh - One-click setup for PSL Analyzer automation

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'
BOLD='\033[1m'

echo ""
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}${BOLD}  PSL Analyzer - Quick Setup${NC}"
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo -e "${YELLOW}⚠ Warning: Run this from project root (face-analyzer-fullstack/)${NC}"
    echo ""
    echo "Current directory: $(pwd)"
    echo ""
    echo "Expected structure:"
    echo "  face-analyzer-fullstack/"
    echo "  ├── backend/"
    echo "  ├── frontend/"
    echo "  └── quick_setup.sh (this script)"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ Found project directories${NC}"
echo ""

# Step 1: Make scripts executable
echo "📋 Step 1: Making scripts executable..."
chmod +x test_all_faces.py 2>/dev/null || echo "  (test_all_faces.py not found yet)"
chmod +x server_manager.sh 2>/dev/null || echo "  (server_manager.sh not found yet)"
echo -e "${GREEN}✓ Scripts are executable${NC}"
echo ""

# Step 2: Create directories
echo "📁 Step 2: Creating directories..."
mkdir -p Face_samples
mkdir -p test_results
mkdir -p logs
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Step 3: Check backend setup
echo "🔧 Step 3: Checking backend..."
cd backend

if [ ! -d "venv" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "  Installing dependencies..."
    pip install -r requirements.txt --quiet
    echo -e "${GREEN}✓ Backend virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Backend virtual environment exists${NC}"
fi

cd ..
echo ""

# Step 4: Check frontend setup
echo "⚛️  Step 4: Checking frontend..."
cd frontend

if [ ! -d "node_modules" ]; then
    echo "  Installing npm dependencies (this may take a minute)..."
    npm install --silent
    echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Frontend dependencies exist${NC}"
fi

cd ..
echo ""

# Step 5: Check ngrok
echo "🌐 Step 5: Checking ngrok..."
if command -v ngrok &> /dev/null; then
    echo -e "${GREEN}✓ ngrok is installed${NC}"
    NGROK_VERSION=$(ngrok version | head -1)
    echo "  Version: $NGROK_VERSION"
else
    echo -e "${YELLOW}⚠ ngrok is not installed${NC}"
    echo ""
    echo "  Install ngrok for public URL sharing:"
    echo "    brew install ngrok/ngrok/ngrok"
    echo "  Or download from: https://ngrok.com/download"
    echo ""
    echo "  After installing, configure your authtoken:"
    echo "    ngrok config add-authtoken YOUR_TOKEN"
    echo "  Get your token at: https://dashboard.ngrok.com/get-started/your-authtoken"
fi
echo ""

# Step 6: Create example Face_samples structure
echo "📸 Step 6: Creating example Face_samples structure..."
if [ ! -d "Face_samples/Example" ]; then
    mkdir -p Face_samples/Example/front
    mkdir -p Face_samples/Example/side
    
    cat > Face_samples/README.md << 'EOF'
# Face_samples Directory

Add your test images here with the following structure:

```
Face_samples/
├── PersonName1/
│   ├── front/
│   │   ├── front1.jpg
│   │   └── front2.jpg
│   └── side/
│       ├── side1.jpg
│       └── side2.jpg
├── PersonName2/
│   ├── front/
│   └── side/
└── ...
```

Each person folder should contain:
- `front/` - Front-facing photos
- `side/` - Side profile photos

The testing script will create all combinations (Cartesian product).
Example: 2 front × 3 side = 6 test pairs

Supported formats: .jpg, .jpeg, .png, .webp
EOF
    
    echo -e "${GREEN}✓ Created example structure and README${NC}"
else
    echo -e "${GREEN}✓ Face_samples directory exists${NC}"
fi
echo ""

# Summary
echo ""
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}${BOLD}  ✅ Setup Complete!${NC}"
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BOLD}Next Steps:${NC}"
echo ""
echo "1. Add test images to Face_samples/:"
echo "   Face_samples/PersonName/front/*.jpg"
echo "   Face_samples/PersonName/side/*.jpg"
echo ""
echo "2. Run automated testing:"
echo "   ${GREEN}python3 test_all_faces.py${NC}"
echo ""
echo "3. Start servers + get public URL:"
echo "   ${GREEN}./server_manager.sh start${NC}"
echo ""
echo "4. Check what's running:"
echo "   ${GREEN}./server_manager.sh status${NC}"
echo ""
echo "5. Stop everything:"
echo "   ${GREEN}./server_manager.sh stop${NC}"
echo ""
echo -e "${BOLD}Documentation:${NC}"
echo "  Read AUTOMATION_GUIDE.md for full details"
echo ""

if ! command -v ngrok &> /dev/null; then
    echo -e "${YELLOW}${BOLD}⚠ Remember to install ngrok for public sharing!${NC}"
    echo ""
fi
