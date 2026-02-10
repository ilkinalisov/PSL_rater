import React, { useMemo, useState } from 'react';

// PART B3: Overlay toggle for original vs analyzed image.
const OverlayViewer = ({ title, originalSrc, overlayData }) => {
  const [mode, setMode] = useState('analysis');
  const overlaySrc = useMemo(() => (overlayData ? `data:image/png;base64,${overlayData}` : ''), [overlayData]);

  if (!overlaySrc && !originalSrc) return null;

  const displayed = mode === 'analysis' ? (overlaySrc || originalSrc) : (originalSrc || overlaySrc);

  return (
    <section className="overlay-viewer-card">
      <div className="overlay-viewer-header">
        <h4>{title}</h4>
        <div className="overlay-toggle-group">
          <button
            type="button"
            className={`overlay-toggle-btn ${mode === 'analysis' ? 'active' : ''}`}
            onClick={() => setMode('analysis')}
          >
            Show Analysis
          </button>
          <button
            type="button"
            className={`overlay-toggle-btn ${mode === 'original' ? 'active' : ''}`}
            onClick={() => setMode('original')}
          >
            Show Original
          </button>
        </div>
      </div>

      <div className="overlay-image-frame">
        <img src={displayed} alt={`${title} ${mode}`} className="overlay-image unified-analysis-image" />
      </div>
    </section>
  );
};

export default OverlayViewer;
