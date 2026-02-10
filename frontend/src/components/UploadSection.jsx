// src/components/UploadSection.jsx
import React, { useState } from 'react';
import '../styles/main.css';

const UploadSection = ({
  frontImage,
  sideImage,
  onFrontImageChange,
  onSideImageChange,
  onAnalyze,
  loading,
  error
}) => {
  const [cameraMode, setCameraMode] = useState(false);
  const [stream, setStream] = useState(null);


  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      setStream(mediaStream);
      setCameraMode(true);
    } catch (err) {
      alert('Error accessing camera: ' + err.message);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
      setCameraMode(false);
    }
  };

  const captureImage = () => {
    const video = document.getElementById('camera-preview');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to blob and use as front image
    canvas.toBlob((blob) => {
      const file = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
      // Create a synthetic event object
      const syntheticEvent = {
        target: {
          files: [file]
        }
      };
      onFrontImageChange(syntheticEvent);
      stopCamera();
    }, 'image/jpeg');
  };

  return (
    <div className="upload-section">
      <h2>Upload Images for Analysis</h2>

      {error && <div className="error-message">{error}</div>}

      <div className="upload-area">
        <div className="image-pair">
          <div className="upload-box">
            <h3>Front Image</h3>
            <input
              type="file"
              accept="image/*"
              onChange={onFrontImageChange}
              className="file-input"
              id="front-upload"
            />
            <label htmlFor="front-upload" className="upload-label">
              {frontImage ? (
                <div className="image-preview">
                  <img src={frontImage} alt="Front preview" />
                </div>
              ) : (
                <div className="upload-placeholder">
                  <span>📷</span>
                  <p>Click to upload front image</p>
                </div>
              )}
            </label>
          </div>

          <div className="upload-box">
            <h3>Side Image</h3>
            <input
              type="file"
              accept="image/*"
              onChange={onSideImageChange}
              className="file-input"
              id="side-upload"
            />
            <label htmlFor="side-upload" className="upload-label">
              {sideImage ? (
                <div className="image-preview">
                  <img src={sideImage} alt="Side preview" />
                </div>
              ) : (
                <div className="upload-placeholder">
                  <span>📷</span>
                  <p>Click to upload side image</p>
                </div>
              )}
            </label>
          </div>
        </div>

        <button
          onClick={onAnalyze}
          className="analyze-btn"
          disabled={loading || !frontImage || !sideImage}
        >
          {loading ? 'Analyzing...' : 'Analyze Images'}
        </button>
      </div>

      <div className="camera-section">
        <h3>Or Use Camera</h3>
        {!cameraMode ? (
          <button onClick={startCamera} className="camera-btn">
            📸 Start Camera
          </button>
        ) : (
          <div className="camera-container">
            <video
              id="camera-preview"
              autoPlay
              ref={(video) => {
                if (video && stream) {
                  video.srcObject = stream;
                }
              }}
            />
            <div className="camera-controls">
              <button onClick={captureImage} className="capture-btn">
                Capture
              </button>
              <button onClick={stopCamera} className="stop-btn">
                Stop
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadSection;