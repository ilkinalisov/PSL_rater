# Facial PSL Analyzer - Project Documentation

> Last Updated: 2026-02-07 (UI Enhanced)

## Project Overview

A fullstack web application that analyzes facial features and calculates PSL (Physical Scale of Looks) scores using computer vision. The application supports both front-facing and side-profile image analysis, providing detailed measurements and scoring based on facial proportions, symmetry, and golden ratio principles.

## Tech Stack

### Backend
- **Framework**: FastAPI 0.104.1
- **Server**: Uvicorn 0.24.0
- **Computer Vision**: OpenCV 4.8.1.78, MediaPipe 0.10.7
- **Data Processing**: NumPy 1.25.2, SciPy 1.11.3, Pandas 2.1.0
- **Image Processing**: Pillow 10.1.0
- **Configuration**: python-dotenv 1.0.0
- **File Handling**: python-multipart 0.0.6

### Frontend
- **Framework**: React 18.2.0
- **HTTP Client**: Axios 1.6.0
- **UI Components**: Material-UI (@mui/material 5.14.18)
- **Styling**: Styled Components 6.1.0, Emotion
- **Visualization**: Plotly.js 2.24.3, react-plotly.js 2.6.0
- **File Upload**: react-dropzone 14.2.3
- **Build Tool**: React Scripts 5.0.1

## Project Structure

```
face-analyzer-fullstack/
├── backend/                      # Python FastAPI backend
│   ├── __init__.py              # Package initialization
│   ├── app.py                   # Main FastAPI application (v2.1.0)
│   ├── facial_analysis.py       # Front-view facial analysis logic
│   ├── side_profile_analyzer.py # Side-profile analysis logic
│   └── requirements.txt         # Python dependencies
├── frontend/                     # React frontend
│   ├── public/                  # Static assets
│   ├── src/
│   │   ├── components/          # React components
│   │   │   ├── UploadSection.jsx
│   │   │   ├── ResultsCard.jsx
│   │   │   ├── OverlayViewer.jsx
│   │   │   └── MetricsDisplay.jsx
│   │   ├── styles/              # CSS stylesheets
│   │   │   ├── main.css
│   │   │   └── ResultsCard.css
│   │   ├── App.jsx              # Main React component
│   │   └── index.js             # React entry point
│   ├── package.json             # Node dependencies
│   └── package-lock.json
├── docs/                        # Documentation
├── exports/                     # Export directory
├── models/                      # ML models (if any)
├── sample_images/              # Sample test images
├── samples/                    # Additional samples
├── .env                        # Environment variables (empty)
├── .gitignore                  # Git ignore rules (empty)
└── README.md                   # Project readme (empty)
```

## Core Features

### 1. Front-View Analysis (facial_analysis.py)
Uses MediaPipe Face Mesh (468 landmarks) to detect and measure:

**Measurements Captured:**
- Face dimensions (width, height, jaw width)
- Eye metrics (canthal tilt, interpupillary distance, separation ratio)
- Nose measurements (width, length, nasal index)
- Mouth and philtrum measurements
- Jaw metrics (jaw-to-cheek ratio, ramus length)
- Facial ratios (fWHR, midface ratio, chin-to-philtrum)
- Vertical proportions (upper/middle/lower thirds)
- Facial symmetry score (0-100%)
- Golden ratio (phi) deviation

**Scoring Components (Weighted):**
- Symmetry: 25%
- Proportions: 20%
- Eyes: 15%
- Jawline: 15%
- Harmony: 15%
- Golden Ratio: 10%

### 2. Side-Profile Analysis (side_profile_analyzer.py)
Analyzes profile angles and projections:

**Measurements Captured:**
- Facial convexity angle (ideal: 165°)
- Nasofrontal angle (ideal: 130°)
- Nasolabial angle (ideal: 95°)
- Gonial angle (ideal: 120°)
- Nasofacial angle (ideal: 36°)
- Forehead slope
- Nasal, chin, and lip projection
- Profile harmony score
- Vertical profile balance

**Scoring Components (Weighted):**
- Profile Harmony: 40%
- Facial Convexity: 20%
- Nasolabial Angle: 15%
- Gonial Angle: 15%
- Vertical Balance: 10%

### 3. Combined Analysis
When both front and side images are provided:
- Front view contributes 70% to final score
- Side view contributes 30% to final score
- Combined score range: 1-10 (PSL scale)

## API Endpoints

### Base URL: `http://localhost:8000`

1. **GET /** - Root endpoint with API info
2. **GET /health** - Health check
3. **POST /analyze** - Legacy endpoint (front view only)
4. **POST /analyze/pair** - Analyze front + side images
5. **POST /analyze/front** - Front view only
6. **POST /analyze/side** - Side profile only
7. **POST /analyze/capture** - Camera capture analysis

### API Response Structure

```json
{
  "success": true,
  "score": {
    "psl": 7.2,
    "interpretation": "Very Good",
    "breakdown": { /* component scores */ },
    "front_contribution": 0.7,
    "side_contribution": 0.3
  },
  "measurements": {
    "front": { /* detailed measurements */ },
    "side": { /* detailed measurements */ },
    "combined": { /* merged measurements */ }
  },
  "overlays": {
    "front": "base64_encoded_image",
    "side": "base64_encoded_image"
  },
  "debug": {
    "front": { /* debug info */ },
    "side": { /* debug info */ }
  },
  "analysis_id": "uuid",
  "timestamp": "ISO8601"
}
```

### PSL Score Interpretation
- 9.0+: Exceptional
- 8.0-8.9: Excellent
- 7.0-7.9: Very Good
- 6.0-6.9: Good
- 5.0-5.9: Average
- 4.0-4.9: Below Average
- 3.0-3.9: Poor
- <3.0: Very Poor

## Frontend Architecture

### Main Components

1. **App.jsx** - Root component managing state
   - Handles API communication
   - Manages loading/error states
   - Orchestrates child components

2. **UploadSection.jsx** - File upload interface
   - Drag-and-drop support (react-dropzone)
   - Single or dual image upload (front + side)
   - Camera capture support

3. **ResultsCard.jsx** - Score visualization
   - PSL score display
   - Score interpretation
   - Visual indicators

4. **OverlayViewer.jsx** - Image overlay display
   - Shows analyzed images with landmarks
   - Supports multiple views (front/side)

5. **MetricsDisplay.jsx** - Detailed measurements
   - Tabular measurement display
   - Score breakdown visualization
   - Plotly charts for data visualization

## Setup Instructions

### Backend Setup

```bash
cd backend

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
# Or with uvicorn directly:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Server will start at: `http://localhost:8000`

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

Application will open at: `http://localhost:3000`

## CORS Configuration

Backend allows requests from:
- `http://localhost:3000`
- `http://127.0.0.1:3000`
- `http://localhost:3001`

**Location**: backend/app.py:386-391

## Key Algorithms

### Canthal Tilt Calculation
```python
angle = atan2(-(outer_y - inner_y), (outer_x - inner_x))
# Positive = upward tilt, Negative = downward tilt
# Ideal range: 5-10°
```

### Facial Symmetry
- Mirrors right-side landmarks across vertical midline
- Computes distance to corresponding left-side landmarks
- Normalizes by face width
- Converts to 0-100% score

### Golden Ratio Deviation
- Compares facial proportions to Phi (1.618...)
- Checks: face width/height, facial thirds ratios
- Returns average deviation from ideal

### Non-linear Score Scaling
- Applies sigmoid-like curve to raw scores
- Prevents extreme scores
- Maps 0-10 input to realistic PSL range

## Image Processing Pipeline

1. **Image Reception**: Uploaded as multipart/form-data or base64
2. **Decoding**: OpenCV decodes to numpy array
3. **Face Detection**: MediaPipe Face Mesh processes image
4. **Landmark Extraction**: 468 landmarks mapped to pixel coordinates
5. **Measurement Calculation**: Distances, angles, ratios computed
6. **Score Calculation**: Weighted scoring across components
7. **Overlay Generation**: Visual annotations on original image
8. **Response Encoding**: Images encoded to base64, JSON response

## Dependencies Notes

### Critical Dependencies
- **MediaPipe**: Core facial landmark detection
- **OpenCV**: Image processing and manipulation
- **NumPy**: Numerical computations
- **FastAPI**: High-performance async web framework
- **React**: Modern UI framework

### Version Constraints
- Python 3.8+ required (for FastAPI)
- Node.js 14+ required (for React 18)

## Known Issues & Limitations

1. **Gonial Angle**: Front-view gonial angle is None (requires side profile)
2. **Landmark Approximation**: Some landmarks approximated if not detected
3. **Single Face**: Analyzes only first detected face
4. **Image Quality**: Requires clear, well-lit images for accuracy
5. **File Size**: No explicit file size limits (may cause memory issues with large images)

## Development Notes

### Code Structure Observations
- **Duplicate Code**: app.py contains duplicate endpoint definitions (lines 1-363 and 365-689)
- **Legacy Support**: Both `/analyze` and `/analyze/front` endpoints for backward compatibility
- **Error Handling**: Comprehensive try-catch blocks with HTTP exceptions
- **Type Safety**: Uses dataclasses for measurement objects with to_dict() serialization

### Potential Improvements
- Remove duplicate code sections in app.py
- Add input validation for file sizes
- Implement rate limiting
- Add caching for repeated analyses
- Database integration for storing results
- User authentication system
- Batch processing support

## Environment Variables

Currently, `.env` file is empty. Potential variables to add:
- `API_PORT`: Backend port (default: 8000)
- `FRONTEND_URL`: Frontend URL for CORS
- `DEBUG_MODE`: Enable/disable debug logging
- `MAX_FILE_SIZE`: Maximum upload size
- `MIN_DETECTION_CONFIDENCE`: MediaPipe confidence threshold

## Testing

### Manual Testing
- Sample images available in `sample_images/` directory
- Use frontend upload or direct API calls with curl/Postman

### Test Cases to Verify
1. Single front image analysis
2. Single side image analysis
3. Paired front+side analysis
4. Camera capture
5. Invalid image format
6. No face detected
7. Multiple faces (should use first face)

## Git Status

- **Repository Initialized**: No (not a git repository)
- **Recommendation**: Initialize git with `git init`
- Update `.gitignore` with common patterns

### Suggested .gitignore
```
# Python
__pycache__/
*.py[cod]
venv/
*.env

# Node
node_modules/
build/
.DS_Store

# IDE
.vscode/
.idea/
*.swp

# Project specific
exports/
sample_images/*.png
sample_images/*.jpg
```

## Maintenance & Updates

### When Adding Features
1. Update this CLAUDE.md with:
   - New dependencies
   - New endpoints
   - Algorithm changes
   - Breaking changes
2. Update component documentation
3. Add to "Known Issues" if applicable

### When Fixing Bugs
1. Document the fix in relevant section
2. Update "Known Issues" if resolved
3. Note any breaking changes

## UI/UX Design System

### Color Palette
- **Primary**: #4F46E5 (Indigo) - Main brand color
- **Primary Dark**: #3730A3 - Hover states
- **Success**: #10B981 (Emerald) - Positive metrics
- **Warning**: #F59E0B (Amber) - Caution indicators
- **Danger**: #EF4444 (Red) - Poor metrics
- **Text Primary**: #111827 (Gray-900) - Main content (AA+ contrast)
- **Text Secondary**: #374151 (Gray-700) - Supporting text
- **Text Muted**: #6B7280 (Gray-500) - Less important text
- **Background**: Linear gradient (Purple to Indigo)
- **Card Background**: #FFFFFF (White)

### Typography
- **Font Family**: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto'
- **Heading Weights**: 700-900 (Bold to Black)
- **Body Weights**: 500-600 (Medium to Semibold)
- **Letter Spacing**: 0.5px - 2px for emphasis

### Accessibility Features
- **Contrast Ratios**: All text meets WCAG AA standards (minimum 4.5:1)
- **Focus Indicators**: 3px solid outline with 2px offset
- **Reduced Motion**: Respects `prefers-reduced-motion` setting
- **High Contrast Mode**: Enhanced visibility in high contrast mode
- **Keyboard Navigation**: Full keyboard support with visible focus states

### Component Styling

#### Upload Section
- Rounded corners: 20px
- Gradient backgrounds on hover
- Drag-and-drop visual feedback
- 3D shadow effects
- Smooth transitions (0.3s cubic-bezier)

#### Score Display
- Large score number: 6rem font size (responsive)
- Color-coded by score range:
  - 9+: Emerald (#10B981)
  - 8-8.9: Blue (#3B82F6)
  - 7-7.9: Purple (#8B5CF6)
  - 6-6.9: Amber (#F59E0B)
  - 5-5.9: Pink (#EC4899)
  - 4-4.9: Red (#EF4444)
  - <4: Dark Red (#DC2626)
- Gradient background matching score color
- 3D hover effects

#### Metric Cards
- Left border accent (6px) matching status color
- Gradient backgrounds
- Hover animations (translate + scale)
- Visual indicators (dots, bars, icons)
- Ideal range visualization

#### Buttons
- Gradient backgrounds
- 3D shadow effects
- Transform on hover (-3px translateY)
- Uppercase text with letter spacing
- Disabled state styling

### Responsive Breakpoints
- **Desktop**: 1024px+
- **Tablet**: 768px - 1023px
- **Mobile**: < 768px
- **Small Mobile**: < 480px

### Animation System
- **Duration**: 0.3s - 0.5s
- **Easing**: cubic-bezier(0.4, 0, 0.2, 1)
- **Fade In**: opacity + translateY
- **Slide In**: opacity + translateX
- **Hover**: translateY + scale
- **Shimmer**: Gradient animation on progress bars

### Version History
- **v2.1.1** (Current): Enhanced UI with improved contrast, visual hierarchy, and accessibility
  - Updated color system with CSS variables
  - Improved text contrast (AA+ compliance)
  - Enhanced score displays with gradients and animations
  - Better responsive design
  - Added focus states for keyboard navigation
  - Implemented dark mode support
  - Added accessibility features (reduced motion, high contrast)
- **v2.1.0**: Full front+side analysis with comprehensive scoring
- **v2.0.0**: Added side profile analysis
- **v1.x**: Initial front-view only analysis

## Contact & Support

Project maintained for personal use. For issues or questions, refer to code comments and this documentation.

---

*This documentation is a living document. Update it whenever significant changes are made to the codebase.*
