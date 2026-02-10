# PSL Analyzer - Complete Automation Guide

## 🎯 Overview

This guide covers two automation solutions:
1. **Automated Testing** - Test all Face_samples pairs automatically
2. **Server Management** - Easy server startup + ngrok public sharing

---

## 📦 Installation

### Step 1: Place the Scripts

```bash
# From your project root (face-analyzer-fullstack/)
cd ~/face-rater-app/face-analyzer-fullstack

# Make scripts executable
chmod +x test_all_faces.py
chmod +x server_manager.sh
```

### Step 2: Install ngrok (for public sharing)

```bash
# Option 1: Homebrew (macOS)
brew install ngrok/ngrok/ngrok

# Option 2: Download from https://ngrok.com/download

# Verify installation
ngrok version
```

### Step 3: Setup ngrok Account (FREE)

1. Sign up at https://dashboard.ngrok.com/signup
2. Get your authtoken
3. Configure it:
   ```bash
   ngrok config add-authtoken YOUR_AUTH_TOKEN
   ```

---

## 🚀 SOLUTION 1: Automated Testing

Tests all image pairs in `Face_samples/` and generates reports.

### Quick Start

```bash
# Run from project root
python3 test_all_faces.py
```

### What It Does

1. ✅ Kills existing processes on ports 8000, 3000
2. ✅ Starts backend server
3. ✅ Discovers all image pairs:
   - AmberHeard: 1 front × 3 side = 3 pairs
   - JordanBarret: 1 front × 6 side = 6 pairs
   - etc.
4. ✅ Tests each pair via API
5. ✅ Saves results:
   - JSON files (raw data)
   - Markdown reports (human-readable)
   - Summary comparison report
   - Overlay images

### Output Structure

```
test_results/
└── test_20260208_143022/
    ├── SUMMARY.md                    # Overall rankings & comparisons
    ├── AmberHeard/
    │   ├── amber_beauty.json
    │   ├── amber_beauty.md
    │   ├── amber_download.json
    │   └── amber_download.md
    ├── HenryCavill/
    │   ├── original_01ce6d74.json
    │   └── original_01ce6d74.md
    └── JordanBarret/
        ├── 2040764_48c7b678.json
        ├── 2040764_48c7b678.md
        └── ... (6 pairs total)
```

### Example Output (SUMMARY.md)

```markdown
# PSL Analyzer Test Results Summary

## Rankings by Overall PSL Score

1. **JordanBarret** - 9.2/10 (Model Tier)
   - Front: 9.3/10, Side: 9.1/10
   - Images: `2040764-500w.jpg` + `images.jpeg`

2. **HenryCavill** - 8.4/10 (High Tier Normie)
   - Front: 8.2/10, Side: 8.6/10
   - Images: `original-2075.webp` + `01ce6d74.jpg`

3. **AmberHeard** - 8.1/10 (High Tier Normie)
   - Front: 7.9/10, Side: 8.3/10
   - Images: `amber-heard.jpg` + `beauty-2013.webp`
```

### Advanced Usage

```python
# Edit configuration at top of test_all_faces.py
FACE_SAMPLES_DIR = "Face_samples"    # Your image folder
RESULTS_DIR = "test_results"         # Output location
BACKEND_PORT = 8000                  # Backend port
```

---

## 🌐 SOLUTION 2: Server Management + Public Sharing

One command to start everything and get a public URL.

### Commands

```bash
# Start everything (backend + frontend + ngrok)
./server_manager.sh start

# Check status
./server_manager.sh status

# View logs
./server_manager.sh logs

# Stop everything
./server_manager.sh stop

# Restart everything
./server_manager.sh restart
```

### What `start` Does

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Starting Backend Server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Killed process on port 8000
ℹ Starting uvicorn on port 8000...
ℹ Using virtual environment
ℹ Waiting for backend to be ready
.........
✓ Backend server started successfully (PID: 12345)
ℹ Backend URL: http://localhost:8000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Starting Frontend Server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Killed process on port 3000
ℹ Starting React development server on port 3000...
ℹ Waiting for frontend to be ready (this may take a minute)
.............
✓ Frontend server started successfully (PID: 12346)
ℹ Frontend URL: http://localhost:3000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Starting ngrok Tunnel
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ℹ Creating public tunnel to http://localhost:3000...
✓ ngrok tunnel created successfully!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🌐 PUBLIC URL (Share this link):
  https://a1b2c3d4.ngrok.io
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ℹ ngrok dashboard: http://localhost:4040
ℹ Anyone can access your app at the public URL above

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🎉 All Services Running!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Local Access:
  Frontend: http://localhost:3000
  Backend:  http://localhost:8000

Public Access (Share with others):
  https://a1b2c3d4.ngrok.io

Management:
  View logs:    tail -f logs/*.log
  ngrok status: http://localhost:4040
  Stop all:     ./server_manager.sh stop
```

### Sharing with Friends

1. **Start the server:**
   ```bash
   ./server_manager.sh start
   ```

2. **Copy the public URL:**
   ```
   https://a1b2c3d4.ngrok.io
   ```

3. **Share it with friends:**
   - Send via text/email/slack
   - They can access from anywhere
   - Works on their phones too!

4. **Monitor usage:**
   - Visit http://localhost:4040
   - See all requests in real-time

### Individual Service Control

```bash
# Start only backend
./server_manager.sh backend

# Start only frontend
./server_manager.sh frontend

# Start only ngrok (requires frontend running)
./server_manager.sh ngrok
```

---

## 📊 Complete Workflow Examples

### Example 1: Test All Faces, Then Share

```bash
# 1. Run automated testing
python3 test_all_faces.py

# 2. Review results
cat test_results/test_*/SUMMARY.md

# 3. Start servers and share
./server_manager.sh start

# 4. Send public URL to friends
# https://xyz123.ngrok.io
```

### Example 2: Quick Demo Session

```bash
# Start everything
./server_manager.sh start

# Share URL with team
# They test the app

# Monitor in real-time
open http://localhost:4040  # ngrok dashboard

# When done
./server_manager.sh stop
```

### Example 3: Development Mode

```bash
# Start just backend + frontend (no ngrok)
./server_manager.sh backend
./server_manager.sh frontend

# Develop and test locally
# http://localhost:3000

# When ready to share
./server_manager.sh ngrok
```

---

## 🔧 Troubleshooting

### Issue: "ngrok not found"

```bash
# Install ngrok
brew install ngrok/ngrok/ngrok

# Or download from https://ngrok.com/download
```

### Issue: "Backend failed to start"

```bash
# Check virtual environment
cd backend
source venv/bin/activate
pip install -r requirements.txt

# Try manually
uvicorn app:app --reload --port 8000
```

### Issue: "Frontend won't start"

```bash
# Reinstall dependencies
cd frontend
rm -rf node_modules package-lock.json
npm install

# Try manually
npm start
```

### Issue: "Port already in use"

```bash
# Kill specific port
kill -9 $(lsof -t -i:8000)  # Backend
kill -9 $(lsof -t -i:3000)  # Frontend

# Or use the script (auto-kills)
./server_manager.sh restart
```

### Issue: "ngrok tunnel won't start"

```bash
# Check if authenticated
ngrok config check

# Re-authenticate
ngrok config add-authtoken YOUR_TOKEN

# Check ngrok logs
cat logs/ngrok.log
```

---

## 📁 File Locations

```
face-analyzer-fullstack/
├── test_all_faces.py              # Automated testing script
├── server_manager.sh              # Server management script
├── Face_samples/                  # Your test images
│   ├── AmberHeard/
│   ├── HenryCavill/
│   └── ...
├── test_results/                  # Generated test reports
│   └── test_20260208_*/
├── logs/                          # Server logs
│   ├── backend.log
│   ├── frontend.log
│   └── ngrok.log
├── .backend.pid                   # Process IDs
├── .frontend.pid
├── .ngrok.pid
└── .ngrok_url                     # Current public URL
```

---

## ⚙️ Configuration

### test_all_faces.py

Edit the configuration section:

```python
# At top of file
BACKEND_URL = "http://localhost:8000"
FACE_SAMPLES_DIR = "Face_samples"
RESULTS_DIR = "test_results"
BACKEND_PORT = 8000
FRONTEND_PORT = 3000
```

### server_manager.sh

Edit configuration variables:

```bash
# At top of file
BACKEND_PORT=8000
FRONTEND_PORT=3000
NGROK_REGION="us"  # Change to: us, eu, ap, au, sa, jp, in
```

---

## 💡 Pro Tips

### 1. Run Tests Overnight

```bash
# Start testing and go to sleep
nohup python3 test_all_faces.py > test_output.log 2>&1 &

# Check progress
tail -f test_output.log
```

### 2. Monitor ngrok Traffic

```bash
# Start servers
./server_manager.sh start

# Open ngrok dashboard
open http://localhost:4040

# See every request, response, and timing
```

### 3. Compare Multiple Test Runs

```bash
# Run test 1
python3 test_all_faces.py  # Creates test_results/test_20260208_100000/

# Change images
# Add more images to Face_samples/

# Run test 2
python3 test_all_faces.py  # Creates test_results/test_20260208_110000/

# Compare
diff test_results/test_*/SUMMARY.md
```

### 4. Custom ngrok Domain (Paid)

```bash
# Edit server_manager.sh
# In start_ngrok() function:
ngrok http $FRONTEND_PORT --region=$NGROK_REGION --domain=your-domain.ngrok.app
```

---

## 🎓 Usage Scenarios

### Scenario 1: Research Paper Data Collection

```bash
# Collect test images in Face_samples/
# Run automated testing
python3 test_all_faces.py

# Get results
cat test_results/test_*/SUMMARY.md

# Export metrics to spreadsheet
cat test_results/test_*/*/*.json | jq '...'
```

### Scenario 2: Team Demo

```bash
# Start everything with public URL
./server_manager.sh start

# Copy public URL from output
# https://abc123.ngrok.io

# Send to team in Slack:
# "Check out the PSL Analyzer: https://abc123.ngrok.io"

# Monitor usage on http://localhost:4040

# When done
./server_manager.sh stop
```

### Scenario 3: A/B Testing Different Algorithms

```bash
# Test with algorithm v1
python3 test_all_faces.py
mv test_results/test_* test_results/v1_results/

# Update algorithm in side_profile_analyzer.py

# Test with algorithm v2
python3 test_all_faces.py
mv test_results/test_* test_results/v2_results/

# Compare
diff test_results/v1_results/SUMMARY.md test_results/v2_results/SUMMARY.md
```

---

## 🔒 Security Notes

### ngrok Free Tier Limitations:
- URL changes each time you restart
- 40 connections/minute limit
- Session lasts 2 hours (then reconnects automatically)

### For Production:
- Use ngrok paid plan for custom domain
- Or deploy to Vercel/Heroku/Railway
- Add authentication to the app

### Privacy:
- Anyone with the ngrok URL can access your app
- Don't share sensitive test images publicly
- Use ngrok's IP restrictions (paid plan)

---

## 📞 Quick Reference

```bash
# Automated Testing
python3 test_all_faces.py                    # Run all tests

# Server Management
./server_manager.sh start                    # Start all + ngrok
./server_manager.sh stop                     # Stop everything
./server_manager.sh status                   # Check status
./server_manager.sh logs                     # View logs
./server_manager.sh restart                  # Restart all

# Individual Services
./server_manager.sh backend                  # Backend only
./server_manager.sh frontend                 # Frontend only
./server_manager.sh ngrok                    # ngrok only

# Monitoring
tail -f logs/*.log                           # Watch all logs
open http://localhost:4040                   # ngrok dashboard
./server_manager.sh status                   # Service status
```

---

## ✅ Checklist

**Before First Use:**
- [ ] Place scripts in project root
- [ ] Make scripts executable (`chmod +x`)
- [ ] Install ngrok
- [ ] Configure ngrok auth token
- [ ] Test backend starts: `cd backend && source venv/bin/activate && uvicorn app:app`
- [ ] Test frontend starts: `cd frontend && npm install && npm start`

**For Automated Testing:**
- [ ] Create `Face_samples/` folder structure
- [ ] Add images to `front/` and `side/` subfolders
- [ ] Run `python3 test_all_faces.py`
- [ ] Check `test_results/` for output

**For Public Sharing:**
- [ ] Run `./server_manager.sh start`
- [ ] Copy public URL from output
- [ ] Share URL with friends
- [ ] Monitor on http://localhost:4040
- [ ] Run `./server_manager.sh stop` when done

---

All set! 🎉 You now have:
1. ✅ Automated testing for all image pairs
2. ✅ One-command server startup
3. ✅ Public URL sharing via ngrok
4. ✅ Easy log monitoring
5. ✅ Clean shutdown management
