import React, { useEffect, useRef, useState } from 'react';

const StepIndicator = ({ current, total }) => {
  const pct = Math.max(0, Math.min(100, (current / total) * 100));
  return (
    <div className="step-indicator">
      <div className="step-label">Step {current} of {total}</div>
      <div className="step-track"><div className="step-fill" style={{ width: `${pct}%` }} /></div>
    </div>
  );
};

const GuideOverlay = ({ type }) => {
  if (type === 'front') {
    return (
      <svg className="guide-overlay" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        <ellipse cx="50" cy="52" rx="26" ry="34" className="guide-stroke" />
        <line x1="35" y1="44" x2="65" y2="44" className="guide-stroke-soft" />
      </svg>
    );
  }

  return (
    <svg className="guide-overlay" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
      <path
        d="M30,18 C26,28 28,40 39,48 C44,52 46,57 45,63 C43,73 47,81 58,82 C68,83 74,76 76,67 C77,58 73,50 67,46 C63,43 60,38 60,34 C60,27 55,20 47,18 Z"
        className="guide-stroke"
      />
      <line x1="58" y1="35" x2="63" y2="37" className="guide-stroke-soft" />
      <line x1="53" y1="72" x2="66" y2="69" className="guide-stroke-soft" />
    </svg>
  );
};

// PART B1: Two-step capture component with camera and upload support.
const PhotoCapture = ({
  step,
  preview,
  onCapture,
  onUpload,
  onRetake,
  onNext,
  onBack,
  canNext,
  isLastStep,
  loading,
}) => {
  const [stream, setStream] = useState(null);
  const [activeFacingMode, setActiveFacingMode] = useState('');
  const [cameraError, setCameraError] = useState('');
  const videoRef = useRef(null);

  const stepTitle = step === 1 ? 'Front Photo' : 'Side Profile';
  const stepHint = step === 1
    ? 'Center your face inside the oval guide.'
    : 'Turn sideways and match the profile guide.';

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  useEffect(() => () => {
    if (stream) stream.getTracks().forEach((track) => track.stop());
  }, [stream]);

  const buildCameraPreferences = () => {
    // Prefer front camera for both steps to match expected selfie UX on mobile.
    if (step === 2) {
      return ['user', 'environment'];
    }
    return ['user'];
  };

  const startCamera = async () => {
    setCameraError('');
    const facingPreferences = buildCameraPreferences();

    for (const facingMode of facingPreferences) {
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: facingMode },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
        setStream(mediaStream);
        setActiveFacingMode(facingMode);
        return;
      } catch (err) {
        if (facingMode === facingPreferences[facingPreferences.length - 1]) {
          setCameraError(`Camera unavailable: ${err.message}`);
        }
      }
    }
  };

  const stopCamera = () => {
    if (!stream) return;
    stream.getTracks().forEach((track) => track.stop());
    setStream(null);
    setActiveFacingMode('');
  };

  const captureFromCamera = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    const shouldMirror = activeFacingMode === 'user';
    if (shouldMirror) {
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const filename = step === 1 ? 'front_capture.jpg' : 'side_capture.jpg';
      const file = new File([blob], filename, { type: 'image/jpeg' });
      onCapture(file, URL.createObjectURL(file));
      stopCamera();
    }, 'image/jpeg', 0.95);
  };

  const handleUpload = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    onUpload(file, URL.createObjectURL(file));
  };

  return (
    <section className="photo-capture-card">
      <StepIndicator current={step} total={2} />
      <h3 className="capture-title">{stepTitle}</h3>
      <p className="capture-subtitle">{stepHint}</p>

      <div className="camera-preview-shell">
        {stream ? (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className={`camera-preview live unified-analysis-image ${activeFacingMode === 'user' ? 'mirrored' : ''}`}
            />
            <GuideOverlay type={step === 1 ? 'front' : 'side'} />
          </>
        ) : preview ? (
          <img src={preview} alt={`${stepTitle} preview`} className="camera-preview still unified-analysis-image" />
        ) : (
          <div className="camera-placeholder">Open camera or upload an image</div>
        )}
      </div>

      {cameraError && <div className="error-banner">{cameraError}</div>}

      <div className="capture-controls">
        {!stream ? (
          <button type="button" className="btn btn-secondary" onClick={startCamera}>Start Camera</button>
        ) : (
          <>
            <button type="button" className="btn btn-primary" onClick={captureFromCamera}>Capture</button>
            <button type="button" className="btn btn-secondary" onClick={stopCamera}>Stop</button>
          </>
        )}

        <label className="btn btn-secondary upload-button" htmlFor={`upload-step-${step}`}>
          Upload File
        </label>
        <input
          id={`upload-step-${step}`}
          type="file"
          accept="image/*"
          onChange={handleUpload}
          className="hidden-input"
        />
      </div>

      <div className="preview-actions">
        {preview && (
          <button type="button" className="btn btn-secondary" onClick={onRetake}>Retake</button>
        )}

        {step > 1 && (
          <button type="button" className="btn btn-secondary" onClick={onBack}>Back</button>
        )}

        {!isLastStep ? (
          <button type="button" className="btn btn-primary" onClick={onNext} disabled={!canNext}>Next Step</button>
        ) : (
          <button type="button" className="btn btn-primary" onClick={onNext} disabled={!canNext || loading}>
            {loading ? 'Analyzing...' : 'Analyze Photos'}
          </button>
        )}
      </div>
    </section>
  );
};

export default PhotoCapture;
