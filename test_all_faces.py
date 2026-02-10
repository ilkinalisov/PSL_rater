#!/usr/bin/env python3
"""
Automated PSL Analyzer Testing Script
Tests all image pairs in Face_samples folder and generates reports
"""

import argparse
import mimetypes
import os
import sys
import time
import json
import math
import requests
from pathlib import Path
from datetime import datetime
import subprocess
from urllib.parse import urlparse

# Configuration
DEFAULT_BACKEND_URL = os.environ.get("BACKEND_URL", "https://looksmaxpsl-rater.onrender.com")
DEFAULT_API_KEY = os.environ.get("API_KEY", "")
DEFAULT_REQUEST_DELAY = float(os.environ.get("REQUEST_DELAY_SECONDS", "11.0"))
FACE_SAMPLES_DIR = "Face_samples"
RESULTS_DIR = "test_results"
BACKEND_PORT = 8000

class Colors:
    """Terminal colors"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run batch PSL tests against local or deployed backend.")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL, help="Backend base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key for X-API-Key header")
    parser.add_argument("--api-version", choices=["v1", "v2"], default="v1", help="Use v1 (/analyze/pair) or v2 (/v2/analyze/pair) endpoint")
    parser.add_argument("--request-delay", type=float, default=DEFAULT_REQUEST_DELAY, help="Delay between successful requests (seconds)")
    parser.add_argument("--force-local-backend", action="store_true", help="Force starting local backend")
    parser.add_argument("--skip-local-backend", action="store_true", help="Never start local backend")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry count for transient errors/rate limits")
    parser.add_argument(
        "--max-same-gonial-ratio",
        type=float,
        default=0.45,
        help="Fail if one exact gonial value dominates more than this ratio."
    )
    return parser.parse_args()

def is_local_url(url):
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        return host in {"localhost", "127.0.0.1", "0.0.0.0"}
    except Exception:
        return False

def kill_process_on_port(port):
    """Kill process running on specified port"""
    try:
        result = subprocess.run(
            f"lsof -t -i:{port}",
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                subprocess.run(f"kill {pid}", shell=True)
            print_success(f"Killed process on port {port}")
            time.sleep(1)
        else:
            print_info(f"No process running on port {port}")
    except Exception as e:
        print_error(f"Error killing process on port {port}: {e}")

def start_backend(backend_url):
    """Start backend server"""
    print_info("Starting backend server...")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print_error("Backend directory not found!")
        return None
    
    # Start uvicorn
    process = subprocess.Popen(
        ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(BACKEND_PORT)],
        cwd=backend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for backend to be ready
    print_info("Waiting for backend to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{backend_url}/health", timeout=1)
            if response.status_code == 200:
                print_success("Backend server started successfully!")
                return process
        except:
            pass
        time.sleep(1)
        print(f"Waiting... ({i+1}/{max_retries})")
    
    print_error("Backend failed to start!")
    return None

def get_image_pairs(base_dir):
    """
    Get all valid front+side image pairs
    Returns: list of tuples (person_name, front_path, side_path)
    """
    pairs = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print_error(f"Directory not found: {base_dir}")
        return pairs
    
    # Iterate through each person's folder
    for person_dir in base_path.iterdir():
        if not person_dir.is_dir():
            continue
        
        person_name = person_dir.name
        front_dir = person_dir / "front"
        side_dir = person_dir / "side"
        
        if not front_dir.exists() or not side_dir.exists():
            print_error(f"Missing front or side folder for {person_name}")
            continue
        
        # Get all images in front and side folders
        front_images = [f for f in front_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
        side_images = [f for f in side_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
        
        if not front_images or not side_images:
            print_error(f"No images found for {person_name}")
            continue
        
        # Create all combinations (Cartesian product)
        for front_img in front_images:
            for side_img in side_images:
                pairs.append((person_name, front_img, side_img))
    
    return pairs

def _build_headers(api_key):
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers

def analyze_pair(front_path, side_path, backend_url, api_key="", max_retries=3, api_version="v1"):
    """Send image pair to API and get results"""
    front_mime = mimetypes.guess_type(front_path.name)[0] or "image/jpeg"
    side_mime = mimetypes.guess_type(side_path.name)[0] or "image/jpeg"
    headers = _build_headers(api_key)

    endpoint_path = "/analyze/pair" if api_version == "v1" else "/v2/analyze/pair"

    for attempt in range(1, max_retries + 1):
        try:
            with open(front_path, 'rb') as front_file, open(side_path, 'rb') as side_file:
                files = {
                    'front_image': (front_path.name, front_file, front_mime),
                    'side_image': (side_path.name, side_file, side_mime)
                }

                response = requests.post(
                    f"{backend_url}{endpoint_path}",
                    files=files,
                    headers=headers,
                    timeout=60
                )

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429 and attempt < max_retries:
                wait_seconds = 12
                print_info(f"Rate limited (429). Waiting {wait_seconds}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_seconds)
                continue

            if response.status_code == 403:
                print_error("API error 403: invalid or missing API key. Use --api-key or set API_KEY env.")
                return None

            detail = None
            try:
                detail = response.json().get("detail") or response.json().get("error")
            except Exception:
                detail = response.text[:200]
            print_error(f"API error: {response.status_code} ({detail})")
            return None

        except Exception as e:
            if attempt < max_retries:
                print_info(f"Request failed ({e}). Retrying {attempt + 1}/{max_retries}...")
                time.sleep(3)
                continue
            print_error(f"Error analyzing pair: {e}")
            return None

    return None


def _extract_front_side_scores(result):
    breakdown = (result.get("overall_score", {}) or {}).get("breakdown", {}) or {}
    front = breakdown.get("front_psl")
    side = breakdown.get("side_psl")
    if front is None:
        front = (result.get("scores_v2", {}) or {}).get("front_primary")
    if side is None:
        side = (result.get("scores_v2", {}) or {}).get("side_primary")
    if front is None:
        front = (result.get("front_analysis", {}) or {}).get("psl")
    if side is None:
        side = (result.get("side_analysis", {}) or {}).get("psl")
    return front, side

def format_result_markdown(person_name, front_name, side_name, result):
    """Format result as markdown"""
    front_psl, side_psl = _extract_front_side_scores(result)
    harmony_bonus = ((result.get("overall_score", {}) or {}).get("breakdown", {}) or {}).get("harmony_bonus", 0)

    md = f"""
## {person_name}

**Images:**
- Front: `{front_name}`
- Side: `{side_name}`

### Overall Score
- **PSL**: {result['overall_score']['psl']}/10
- **Category**: {result['overall_score']['category']}
- **Interpretation**: {result['overall_score']['interpretation']}

### Breakdown
- **Front Score**: {front_psl}/10
- **Side Score**: {side_psl}/10
- **Harmony Bonus**: +{harmony_bonus}

### Key Metrics
- **Gonial Angle**: {result['overall_score']['breakdown']['key_metrics'].get('gonial_angle', 'N/A')}°
- **Canthal Tilt**: {result['overall_score']['breakdown']['key_metrics'].get('canthal_tilt', 'N/A')}°
- **Facial Symmetry**: {result['overall_score']['breakdown']['key_metrics'].get('facial_symmetry', 'N/A')}%
- **Profile Harmony**: {result['overall_score']['breakdown']['key_metrics'].get('profile_harmony', 'N/A')}%

### Front Analysis Details
"""
    
    # Front component scores
    front_components = ((result.get('overall_score', {}) or {}).get('breakdown', {}) or {}).get('front_components')
    if not front_components:
        front_components = (result.get('front_analysis', {}) or {}).get('breakdown', {}) or {}
    for comp, score in front_components.items():
        md += f"- **{comp.replace('_', ' ').title()}**: {score:.1f}/10\n"
    
    md += "\n### Side Analysis Details\n"
    
    # Side component scores
    side_components = ((result.get('overall_score', {}) or {}).get('breakdown', {}) or {}).get('side_components')
    if not side_components:
        side_components = (result.get('side_analysis', {}) or {}).get('breakdown', {}) or {}
    for comp, score in side_components.items():
        md += f"- **{comp.replace('_', ' ').title()}**: {score:.1f}/10\n"
    
    # Side measurements
    side_measurements = result['side_analysis']['measurements']
    md += f"\n### Side Profile Measurements\n"
    md += f"- **Gonial Angle**: {side_measurements.get('gonial_angle', 'N/A')}°\n"
    md += f"- **Nasolabial Angle**: {side_measurements.get('nasolabial_angle', 'N/A')}°\n"
    md += f"- **Facial Convexity**: {side_measurements.get('facial_convexity_angle', 'N/A')}°\n"
    md += f"- **Was Mirrored**: {side_measurements.get('was_mirrored', False)}\n"
    md += f"- **Is Estimated**: {side_measurements.get('is_estimated', False)}\n"
    
    md += "\n" + "="*60 + "\n"
    
    return md

def _extract_gonial(result):
    side = result.get("side_analysis", {}) or {}
    measurements = side.get("measurements", {}) or {}
    if measurements.get("gonial_angle") is not None:
        return float(measurements.get("gonial_angle"))
    return ((result.get("overall_score", {}) or {}).get("breakdown", {}) or {}).get("key_metrics", {}).get("gonial_angle")


def create_summary_report(all_results, output_dir):
    """Create summary comparison report"""
    md = f"""# PSL Analyzer Test Results Summary
    
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Pairs Tested**: {len(all_results)}

---

## Rankings by Overall PSL Score

"""
    
    # Sort by overall PSL score
    sorted_results = sorted(all_results, key=lambda x: x['result']['overall_score']['psl'], reverse=True)
    
    for i, item in enumerate(sorted_results, 1):
        result = item['result']
        front_psl, side_psl = _extract_front_side_scores(result)
        md += f"{i}. **{item['person']}** - {result['overall_score']['psl']}/10 ({result['overall_score']['category']})\n"
        md += f"   - Front: {front_psl}/10, "
        md += f"Side: {side_psl}/10\n"
        md += f"   - Images: `{item['front_name']}` + `{item['side_name']}`\n\n"
    
    md += "\n---\n\n## Metric Comparisons\n\n"
    
    # Gonial angle comparison
    md += "### Gonial Angles\n\n"
    for item in sorted_results:
        gonial = _extract_gonial(item['result'])
        if gonial is not None:
            md += f"- **{item['person']}**: {gonial:.1f}°\n"

    # Gonial distribution diagnostics
    rounded_gonials = []
    for item in sorted_results:
        g = _extract_gonial(item["result"])
        if g is None:
            continue
        rounded_gonials.append(round(float(g), 1))
    diag = {
        "count": len(rounded_gonials),
        "unique_count": 0,
        "unique_ratio": 0.0,
        "max_same_ratio": 0.0,
        "entropy_bits": 0.0,
        "most_common": None,
    }
    if rounded_gonials:
        from collections import Counter
        counts = Counter(rounded_gonials)
        total = float(len(rounded_gonials))
        max_val, max_count = counts.most_common(1)[0]
        entropy = 0.0
        for c in counts.values():
            p = c / total
            entropy -= p * math.log2(max(p, 1e-12))
        diag.update({
            "unique_count": len(counts),
            "unique_ratio": len(counts) / total,
            "max_same_ratio": max_count / total,
            "entropy_bits": entropy,
            "most_common": {"value": max_val, "count": max_count},
        })

    md += "\n### Gonial Distribution Diagnostics\n\n"
    md += f"- Samples: {diag['count']}\n"
    md += f"- Unique gonial values: {diag['unique_count']} ({diag['unique_ratio']:.2f} ratio)\n"
    md += f"- Max same-angle ratio: {diag['max_same_ratio']:.2f}\n"
    md += f"- Shannon entropy: {diag['entropy_bits']:.2f} bits\n"
    if diag["most_common"]:
        md += f"- Most common value: {diag['most_common']['value']:.1f}° ({diag['most_common']['count']} samples)\n"
    
    # Canthal tilt comparison
    md += "\n### Canthal Tilts\n\n"
    for item in sorted_results:
        tilt = item['result']['overall_score']['breakdown']['key_metrics'].get('canthal_tilt')
        if tilt:
            md += f"- **{item['person']}**: {tilt:.1f}°\n"
    
    # Symmetry comparison
    md += "\n### Facial Symmetry\n\n"
    for item in sorted_results:
        symmetry = item['result']['overall_score']['breakdown']['key_metrics'].get('facial_symmetry')
        if symmetry:
            md += f"- **{item['person']}**: {symmetry:.1f}%\n"
    
    # Save summary
    summary_path = output_dir / "SUMMARY.md"
    with open(summary_path, 'w') as f:
        f.write(md)
    
    print_success(f"Summary report saved: {summary_path}")
    return diag

def main():
    args = parse_args()
    backend_url = args.backend_url.strip().rstrip("/")
    api_key = args.api_key.strip()
    api_version = args.api_version.strip()

    print_header("PSL Analyzer - Automated Testing")
    print_info(f"Backend URL: {backend_url}")
    print_info(f"API version: {api_version}")
    if api_key:
        print_info("API key: configured")
    else:
        print_info("API key: not set")
    
    # Create results directory
    results_dir = Path(RESULTS_DIR)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = results_dir / f"test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_info(f"Results will be saved to: {output_dir}")
    
    should_start_local = (
        not args.skip_local_backend and
        (args.force_local_backend or is_local_url(backend_url))
    )

    backend_process = None
    if should_start_local:
        # Step 1: Kill existing local backend process
        print_header("Step 1: Cleaning Up Existing Local Backend Process")
        kill_process_on_port(BACKEND_PORT)

        # Step 2: Start backend
        print_header("Step 2: Starting Local Backend Server")
        backend_process = start_backend(backend_url)

        if not backend_process:
            print_error("Failed to start backend. Exiting.")
            sys.exit(1)
    else:
        # Step 1: Check remote backend
        print_header("Step 1: Checking Remote Backend")
        try:
            health = requests.get(f"{backend_url}/health", timeout=10)
            if health.status_code == 200:
                print_success("Remote backend is reachable.")
            else:
                print_error(f"Health check failed: HTTP {health.status_code}")
                sys.exit(1)
        except Exception as exc:
            print_error(f"Could not reach backend health endpoint: {exc}")
            sys.exit(1)

    try:
        # Step 3: Get all image pairs
        print_header("Step 3: Discovering Image Pairs")
        pairs = get_image_pairs(FACE_SAMPLES_DIR)
        
        if not pairs:
            print_error("No image pairs found!")
            return
        
        print_success(f"Found {len(pairs)} image pairs to test")
        
        # Show breakdown by person
        from collections import Counter
        person_counts = Counter(p[0] for p in pairs)
        for person, count in person_counts.items():
            print_info(f"  {person}: {count} pair(s)")
        
        # Step 4: Test all pairs
        print_header("Step 4: Testing All Pairs")
        
        all_results = []
        
        for i, (person, front_path, side_path) in enumerate(pairs, 1):
            print(f"\n{Colors.BOLD}[{i}/{len(pairs)}] Testing: {person}{Colors.ENDC}")
            print(f"  Front: {front_path.name}")
            print(f"  Side:  {side_path.name}")
            
            result = analyze_pair(
                front_path,
                side_path,
                backend_url=backend_url,
                api_key=api_key,
                max_retries=max(1, args.max_retries),
                api_version=api_version
            )
            
            if result:
                psl_score = result['overall_score']['psl']
                category = result['overall_score']['category']
                print_success(f"Score: {psl_score}/10 ({category})")
                
                # Save individual result
                person_dir = output_dir / person
                person_dir.mkdir(exist_ok=True)
                
                # Save JSON
                json_filename = f"{front_path.stem}_{side_path.stem}.json"
                json_path = person_dir / json_filename
                with open(json_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                # Save Markdown
                md_content = format_result_markdown(person, front_path.name, side_path.name, result)
                md_filename = f"{front_path.stem}_{side_path.stem}.md"
                md_path = person_dir / md_filename
                with open(md_path, 'w') as f:
                    f.write(md_content)
                
                all_results.append({
                    'person': person,
                    'front_name': front_path.name,
                    'side_name': side_path.name,
                    'result': result
                })
            else:
                print_error("Analysis failed!")
            
            # Delay between requests to stay under deployed rate limits.
            time.sleep(max(0.0, args.request_delay))
        
        # Step 5: Create summary report
        print_header("Step 5: Generating Summary Report")
        diagnostics = create_summary_report(all_results, output_dir)
        
        print_header("Testing Complete!")
        print_success(f"All results saved to: {output_dir}")
        print_info("\nFiles created:")
        print_info(f"  - Individual results: {output_dir}/<person>/*.json")
        print_info(f"  - Markdown reports: {output_dir}/<person>/*.md")
        print_info(f"  - Summary report: {output_dir}/SUMMARY.md")

        if diagnostics["count"] > 0 and diagnostics["max_same_ratio"] > float(args.max_same_gonial_ratio):
            print_error(
                "Gonial collapse detected: one angle dominates "
                f"{diagnostics['max_same_ratio']:.2f} of samples "
                f"(threshold {float(args.max_same_gonial_ratio):.2f})."
            )
            sys.exit(2)
        
    finally:
        # Cleanup
        print_info("\nCleaning up...")
        if backend_process:
            backend_process.terminate()
            backend_process.wait(timeout=5)

            kill_process_on_port(BACKEND_PORT)
            print_success("Cleanup complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_info("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
