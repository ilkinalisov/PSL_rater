import React, { useMemo, useState } from 'react';

// PART B3: Overlay toggle for original vs analyzed image.
const OverlayViewer = ({ title, originalSrc, overlayData, contours }) => {
  const [mode, setMode] = useState('analysis');
  const overlaySrc = useMemo(() => (overlayData ? `data:image/png;base64,${overlayData}` : ''), [overlayData]);
  const normalizedContours = useMemo(() => {
    if (!contours || typeof contours !== 'object') return null;
    const toPointList = (raw) => {
      if (!Array.isArray(raw)) return [];
      return raw
        .map((pt) => {
          if (!pt) return null;
          if (Array.isArray(pt) && pt.length >= 2) return { x: Number(pt[0]), y: Number(pt[1]) };
          if (typeof pt === 'object' && Number.isFinite(pt.x) && Number.isFinite(pt.y)) return { x: Number(pt.x), y: Number(pt.y) };
          return null;
        })
        .filter((pt) => pt && Number.isFinite(pt.x) && Number.isFinite(pt.y));
    };

    const silhouette = toPointList(contours.silhouette);
    const jawRamus = toPointList(contours.jaw_ramus);
    const width = Number(contours?.debug?.image_size?.width) || 0;
    const height = Number(contours?.debug?.image_size?.height) || 0;
    return {
      method: contours.method || 'unknown',
      confidence: Number(contours.confidence) || 0,
      silhouette,
      jawRamus,
      width,
      height,
      wasMirrored: Boolean(contours?.debug?.was_mirrored),
    };
  }, [contours]);

  if (!overlaySrc && !originalSrc) return null;

  const displayed = mode === 'analysis' ? (overlaySrc || originalSrc) : (originalSrc || overlaySrc);
  const canDrawContours =
    normalizedContours &&
    normalizedContours.width > 0 &&
    normalizedContours.height > 0 &&
    (normalizedContours.silhouette.length >= 2 || normalizedContours.jawRamus.length >= 2);

  const toPolyline = (points, flipX = false) => {
    if (!Array.isArray(points) || points.length < 2) return '';
    const width = normalizedContours?.width || 0;
    return points
      .map((pt) => {
        const x = flipX ? (width - 1 - pt.x) : pt.x;
        return `${x},${pt.y}`;
      })
      .join(' ');
  };
  const flipForOriginal = mode === 'original' && Boolean(normalizedContours?.wasMirrored);

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
        <div className="overlay-stage">
          <img src={displayed} alt={`${title} ${mode}`} className="overlay-image unified-analysis-image" />
          {canDrawContours && (
            <svg
              className="contour-overlay"
              viewBox={`0 0 ${normalizedContours.width} ${normalizedContours.height}`}
              preserveAspectRatio="xMidYMid meet"
              aria-hidden="true"
            >
              {normalizedContours.silhouette.length >= 2 && (
                <polyline
                  points={toPolyline(normalizedContours.silhouette, flipForOriginal)}
                  className="contour-line contour-silhouette"
                />
              )}
              {normalizedContours.jawRamus.length >= 2 && (
                <polyline
                  points={toPolyline(normalizedContours.jawRamus, flipForOriginal)}
                  className="contour-line contour-jaw-ramus"
                />
              )}
            </svg>
          )}
        </div>
      </div>
    </section>
  );
};

export default OverlayViewer;
