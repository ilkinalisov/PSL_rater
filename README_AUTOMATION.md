# 🚀 PSL Analyzer - Automation Quick Start

Two powerful automation tools to save you hours of work!

---

## 🎯 Tool #1: Automated Testing

**Problem:** Manually testing image pairs is tedious  
**Solution:** Test ALL pairs automatically and get reports

### One Command
```bash
python3 test_all_faces.py
```

### What It Does
```
Face_samples/
├── HenryCavill/
│   ├── front/ (1 image)
│   └── side/  (1 image)     → 1×1 = 1 pair tested
├── JordanBarret/
│   ├── front/ (1 image)
│   └── side/  (6 images)    → 1×6 = 6 pairs tested
└── AmberHeard/
    ├── front/ (1 image)
    └── side/  (3 images)    → 1×3 = 3 pairs tested
                               ──────────────────────
                               Total: 10 pairs tested
```

### Output
```
test_results/test_20260208_143022/
├── SUMMARY.md              ← Rankings & comparisons
├── HenryCavill/
│   ├── front_side.json     ← Raw data
│   └── front_side.md       ← Human-readable
├── JordanBarret/
│   ├── front_side1.json
│   ├── front_side1.md
│   ├── front_side2.json
│   └── ... (6 pairs)
└── AmberHeard/
    └── ... (3 pairs)
```

---

## 🌐 Tool #2: Server Manager + Public Sharing

**Problem:** Starting servers manually is repetitive  
**Solution:** One command to start everything + get public URL

### One Command
```bash
./server_manager.sh start
```

### What It Does
```
✓ Kills existing processes
✓ Starts backend  (http://localhost:8000)
✓ Starts frontend (http://localhost:3000)
✓ Creates ngrok tunnel

🌐 PUBLIC URL: https://abc123.ngrok.io
   ↑ Share this with anyone!
```

### Other Commands
```bash
./server_manager.sh status     # Check what's running
./server_manager.sh logs       # View all logs
./server_manager.sh stop       # Stop everything
./server_manager.sh restart    # Restart all
```

---

## 📥 Installation (One-Time Setup)

### Quick Setup (Automated)
```bash
cd ~/face-rater-app/face-analyzer-fullstack

# Run the setup script
chmod +x quick_setup.sh
./quick_setup.sh
```

### Manual Setup
```bash
# 1. Make scripts executable
chmod +x test_all_faces.py
chmod +x server_manager.sh

# 2. Install ngrok (for public sharing)
brew install ngrok/ngrok/ngrok

# 3. Configure ngrok
ngrok config add-authtoken YOUR_TOKEN
# Get token at: https://dashboard.ngrok.com/get-started/your-authtoken

# 4. Install Python dependencies
pip install requests

# Done!
```

---

## 📖 Usage Examples

### Example 1: Test All Faces
```bash
# Add images to Face_samples/
Face_samples/
├── Henry/front/pic.jpg
└── Henry/side/pic.jpg

# Run tests
python3 test_all_faces.py

# View results
cat test_results/test_*/SUMMARY.md
```

### Example 2: Share with Friends
```bash
# Start everything
./server_manager.sh start

# Output shows:
🌐 PUBLIC URL: https://xyz123.ngrok.io

# Send link to friends
# They can access from anywhere!
# Even on their phones!

# When done
./server_manager.sh stop
```

### Example 3: Development Workflow
```bash
# Start servers
./server_manager.sh start

# Code and test locally
# http://localhost:3000

# View logs in real-time
./server_manager.sh logs

# Restart after changes
./server_manager.sh restart
```

---

## 🎬 Complete Workflow

```bash
# 1. Add test images
mkdir -p Face_samples/TestPerson/front
mkdir -p Face_samples/TestPerson/side
# Add images to those folders

# 2. Run automated tests
python3 test_all_faces.py

# 3. Review results
cat test_results/test_*/SUMMARY.md
cat test_results/test_*/TestPerson/*.md

# 4. Share with others
./server_manager.sh start

# 5. Send them the public URL
# https://abc123.ngrok.io

# 6. Stop when done
./server_manager.sh stop
```

---

## 📊 Sample Output

### Automated Testing Output
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PSL Analyzer - Automated Testing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Killed process on port 8000
✓ Backend server started successfully
✓ Found 10 image pairs to test
  HenryCavill: 1 pair(s)
  JordanBarret: 6 pair(s)
  AmberHeard: 3 pair(s)

[1/10] Testing: HenryCavill
  Front: original.webp
  Side:  profile.jpg
✓ Score: 8.4/10 (High Tier Normie)

[2/10] Testing: JordanBarret
  Front: frontal.jpg
  Side:  side1.jpg
✓ Score: 9.2/10 (Model Tier)

...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Testing Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ All results saved to: test_results/test_20260208_143022
```

### Server Manager Output
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Starting Backend Server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Backend server started successfully (PID: 12345)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Starting Frontend Server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Frontend server started successfully (PID: 12346)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Starting ngrok Tunnel
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ ngrok tunnel created successfully!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🌐 PUBLIC URL (Share this link):
  https://a1b2c3d4.ngrok.io
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🎉 All Services Running!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Local Access:
  Frontend: http://localhost:3000
  Backend:  http://localhost:8000

Public Access (Share with others):
  https://a1b2c3d4.ngrok.io
```

---

## 🛠️ Files Included

| File | Purpose |
|------|---------|
| `test_all_faces.py` | Automated testing script |
| `server_manager.sh` | Server management + ngrok |
| `quick_setup.sh` | One-click setup |
| `AUTOMATION_GUIDE.md` | Complete documentation |
| `test_requirements.txt` | Python dependencies |

---

## 🎓 Common Scenarios

### "I want to test multiple celebrity faces"
```bash
# Add images
Face_samples/
├── Cavill/front/ + side/
├── Barrett/front/ + side/
└── Heard/front/ + side/

# Test all
python3 test_all_faces.py

# Compare
cat test_results/test_*/SUMMARY.md
```

### "I want friends to try the app"
```bash
./server_manager.sh start
# Send them: https://xyz.ngrok.io
# They test it from their phones/computers
./server_manager.sh stop
```

### "I'm developing and testing locally"
```bash
./server_manager.sh backend    # Start backend only
./server_manager.sh frontend   # Start frontend only
# Code, test, repeat
./server_manager.sh restart    # Restart after changes
```

---

## 📞 Quick Reference Card

```bash
┌──────────────────────────────────────────────┐
│         AUTOMATED TESTING                    │
├──────────────────────────────────────────────┤
│ python3 test_all_faces.py   # Run all tests │
│                                              │
│ Results in: test_results/test_*/            │
│ - SUMMARY.md (rankings)                      │
│ - PersonName/*.json (raw data)               │
│ - PersonName/*.md (readable)                 │
└──────────────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│         SERVER MANAGEMENT                    │
├──────────────────────────────────────────────┤
│ ./server_manager.sh start    # All services │
│ ./server_manager.sh stop     # Stop all     │
│ ./server_manager.sh status   # Check status │
│ ./server_manager.sh logs     # View logs    │
│ ./server_manager.sh restart  # Restart all  │
└──────────────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│         MONITORING                           │
├──────────────────────────────────────────────┤
│ http://localhost:3000  # Frontend (local)   │
│ http://localhost:8000  # Backend (local)    │
│ http://localhost:4040  # ngrok dashboard    │
│ tail -f logs/*.log     # Watch logs         │
└──────────────────────────────────────────────┘
```

---

## ✅ Checklist

**First Time Setup:**
- [ ] Run `./quick_setup.sh`
- [ ] Install ngrok: `brew install ngrok/ngrok/ngrok`
- [ ] Configure ngrok: `ngrok config add-authtoken TOKEN`
- [ ] Test backend: `./server_manager.sh backend`
- [ ] Test frontend: `./server_manager.sh frontend`

**Before Testing:**
- [ ] Images in `Face_samples/PersonName/front/`
- [ ] Images in `Face_samples/PersonName/side/`
- [ ] Backend has virtual environment
- [ ] Frontend has node_modules

**Before Sharing:**
- [ ] ngrok is configured
- [ ] Backend + frontend tested locally
- [ ] Know your use case (demo, testing, etc.)

---

## 🆘 Help

**Something not working?**
1. Read `AUTOMATION_GUIDE.md` (full documentation)
2. Check logs: `tail -f logs/*.log`
3. Restart: `./server_manager.sh restart`
4. Check status: `./server_manager.sh status`

**Still stuck?**
- Check backend runs: `cd backend && source venv/bin/activate && uvicorn app:app`
- Check frontend runs: `cd frontend && npm start`
- Verify ngrok: `ngrok version` and `ngrok config check`

---

## 🎉 You're Ready!

Everything is set up. Just run:

```bash
# Test all faces
python3 test_all_faces.py

# Share with friends
./server_manager.sh start
```

Enjoy! 🚀
