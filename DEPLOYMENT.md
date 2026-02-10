# Deployment Guide

This project can be deployed cheaply and securely with:
- Frontend: Vercel (free)
- Backend: Render free tier or Railway hobby plan

## 1. Prepare Backend Repo
1. Ensure these files exist at repo root:
   - `Dockerfile`
   - `requirements.txt`
   - `render.yaml`
   - 'supabase, cloudflare'
   - `railway.toml`
2. Backend app entrypoint is `backend/main.py`.

## 2. Deploy Backend on Render (Free)
1. Sign in to Render and create a new Web Service from GitHub.
2. Render detects `render.yaml` and builds from `Dockerfile`.
3. Wait for deploy, then open `https://<your-backend>.onrender.com/health`.
4. Copy backend URL (example: `https://psl-analyzer-api.onrender.com`).

## 3. Deploy Frontend on Vercel (Free)
1. Sign in to Vercel and import your frontend repo/project.
2. Set env var:
   - `REACT_APP_API_URL=https://<your-backend>.onrender.com`
3. Deploy and copy the frontend URL (example: `https://your-app.vercel.app`).

## 4. Wire CORS
1. In backend host settings, set:
   - `FRONTEND_URL=https://your-app.vercel.app`
   - `API_KEY=<shared-secret>`
2. Redeploy backend.

## 4.1 Optional: Dedicated V2 Side Inference Service
1. Deploy `side_inference_service/app.py` on a separate instance (RunPod/Modal/Render).
2. Set backend env vars:
   - `SIDE_INFERENCE_URL=https://<your-side-service>`
   - `V2_SIDE_TIMEOUT_SECONDS=3.2`
   - `V2_CB_FAILURE_THRESHOLD=3`
   - `V2_CB_RECOVERY_SECONDS=20`
   - `ENABLE_V2_SHADOW=true` (optional shadow logging)
3. Optional calibration model:
   - `V2_CALIBRATION_MODEL_PATH=/path/to/calibration_model.json`

## 5. End-to-End Test
1. Open frontend URL.
2. Capture/upload front + side images.
3. Confirm API calls succeed and overlays render.

## 6. Railway Alternative (Lower Latency)
1. Create Railway project from repo.
2. Railway uses `railway.toml` + Dockerfile.
3. Set `FRONTEND_URL` env var in Railway.
4. Deploy and point frontend env to Railway backend URL.

## Notes
- Render free tier has cold starts (can be ~20-30 seconds after idle).
- Railway hobby plan (~$5/mo) reduces/eliminates cold-start delays.
- `/v2/analyze/side` and `/v2/analyze/pair` are now available with `landmarks_v2`, `quality_v2`, `scores_v2`, and edge-traced `contours` (`silhouette`, `jaw_ramus`) for frontend polyline rendering.
