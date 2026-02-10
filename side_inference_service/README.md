# Side Inference Service (V2)

Dedicated side-profile inference service intended for RunPod/Modal style deployment.

## Endpoint

- `POST /infer/side` with multipart image upload field `image`
- `GET /health`
- Health now reports remote engine availability (`3DDFA` optional runtime).

## Run locally

```bash
python3 -m venv .venv-side
source .venv-side/bin/activate
pip install -r side_inference_service/requirements.txt
uvicorn side_inference_service.app:app --host 0.0.0.0 --port 8100
```

## Backend integration

Set in backend environment:

- `SIDE_INFERENCE_URL=http://localhost:8100`
- Optional: `V2_SIDE_TIMEOUT_SECONDS=3.2`
- Optional: `V2_CB_FAILURE_THRESHOLD=3`
- Optional: `V2_CB_RECOVERY_SECONDS=20`

If `SIDE_INFERENCE_URL` is absent or unavailable, backend automatically falls back to local V2 adapter.

## Render deployment

Example service manifest: `side_inference_service/render.yaml`.
