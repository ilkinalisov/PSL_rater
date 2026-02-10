import React, { useRef, useState } from 'react';
import './styles/main.css';
import PhotoCapture from './components/PhotoCapture';
import ResultCard from './components/ResultCard';
import OverlayViewer from './components/OverlayViewer';
import ScoreGauge from './components/ScoreGauge';
import MetricsDisplay from './components/MetricsDisplay';
import { analyzePair as analyzePairRequest } from './api';

function App() {
  const captureRef = useRef(null);
  const [step, setStep] = useState(1);
  const [frontImage, setFrontImage] = useState(null);
  const [sideImage, setSideImage] = useState(null);
  const [frontPreview, setFrontPreview] = useState('');
  const [sidePreview, setSidePreview] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const currentPreview = step === 1 ? frontPreview : sidePreview;

  const scrollToCapture = () => {
    captureRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const resetAll = () => {
    setStep(1);
    setFrontImage(null);
    setSideImage(null);
    setFrontPreview('');
    setSidePreview('');
    setResults(null);
    setError('');
    setLoading(false);
  };

  const handleGetStarted = () => {
    resetAll();
    window.setTimeout(() => {
      scrollToCapture();
    }, 80);
  };

  const handlePhotoSet = (file, preview) => {
    // PART B1: Two-step front/side image assignment.
    if (step === 1) {
      setFrontImage(file);
      setFrontPreview(preview);
      return;
    }
    setSideImage(file);
    setSidePreview(preview);
  };

  const handleRetake = () => {
    if (step === 1) {
      setFrontImage(null);
      setFrontPreview('');
      return;
    }
    setSideImage(null);
    setSidePreview('');
  };

  const analyzePair = async () => {
    if (!frontImage || !sideImage) {
      setError('Please provide both front and side images.');
      return;
    }

    setLoading(true);
    setError('');
    setResults(null);

    try {
      const result = await analyzePairRequest(frontImage, sideImage);
      setResults(result);
    } catch (err) {
      setError(err.message || 'Analysis failed. Please try different photos.');
    } finally {
      setLoading(false);
    }
  };

  const handleNext = () => {
    if (step === 1) {
      if (!frontImage) {
        setError('Please capture or upload a front photo first.');
        return;
      }
      setError('');
      setStep(2);
      return;
    }

    analyzePair();
  };

  return (
    <div className="app-shell">
      <header className="hero">
        <p className="hero-kicker">PSL Facial Analyzer</p>
        <h1 className="hero-title">Analyze your facial proportions with AI</h1>
        <p className="hero-copy">
          Upload or capture front and side photos to estimate facial proportions, profile angles, and combined PSL score.
        </p>
        <button type="button" className="btn btn-primary" onClick={handleGetStarted}>Get Started</button>
      </header>

      {loading && (
        <div className="analysis-loading-screen" role="status" aria-live="polite">
          <div className="analysis-loading-card">
            <div className="analysis-spinner" />
            <h3>Analyzing your photos...</h3>
            <p>Detecting landmarks, building overlays, and calculating PSL scores.</p>
          </div>
        </div>
      )}

      {!results ? (
        <main className="content-wrap" ref={captureRef}>
          <section className="panel">
            <PhotoCapture
              step={step}
              preview={currentPreview}
              onCapture={handlePhotoSet}
              onUpload={handlePhotoSet}
              onRetake={handleRetake}
              onNext={handleNext}
              onBack={() => {
                setError('');
                setStep(1);
              }}
              canNext={step === 1 ? Boolean(frontImage) : Boolean(sideImage && frontImage)}
              isLastStep={step === 2}
              loading={loading}
            />

            <div className="capture-summary">
              <div className={`capture-chip ${frontImage ? 'ready' : ''}`}>Front: {frontImage ? 'Ready' : 'Missing'}</div>
              <div className={`capture-chip ${sideImage ? 'ready' : ''}`}>Side: {sideImage ? 'Ready' : 'Missing'}</div>
            </div>

            {error && <div className="error-banner">{error}</div>}
          </section>
        </main>
      ) : (
        <main className="content-wrap fade-up">
          <section className="panel overall-panel">
            {/* PART B2: Main score visual with high readability and color coding. */}
            <div className="overall-head">
              <div>
                <h2>Overall PSL Score</h2>
                <p className="muted">{results.overall_score?.interpretation}</p>
              </div>
              <button type="button" className="btn btn-primary new-analysis-btn" onClick={handleGetStarted}>
                New Analysis
              </button>
            </div>

            <div className="overall-body">
              <ScoreGauge score={results.overall_score?.psl || 0} label={results.overall_score?.category || 'PSL'} />
              <div className="overall-meta">
                <div className="meta-row"><span>Category</span><strong>{results.overall_score?.category}</strong></div>
                <div className="meta-row"><span>Front View</span><strong>{results.overall_score?.breakdown?.front_psl?.toFixed?.(1) ?? results.overall_score?.breakdown?.front_psl}</strong></div>
                <div className="meta-row"><span>Side Profile</span><strong>{results.overall_score?.breakdown?.side_psl?.toFixed?.(1) ?? results.overall_score?.breakdown?.side_psl}</strong></div>
                <div className="meta-row"><span>Harmony Bonus</span><strong>+{results.overall_score?.breakdown?.harmony_bonus?.toFixed?.(2) ?? results.overall_score?.breakdown?.harmony_bonus}</strong></div>
                <div className="meta-row"><span>Gender</span><strong>{results.gender?.label || 'unknown'} ({(results.gender?.confidence ?? 0).toFixed(2)})</strong></div>
              </div>
            </div>
          </section>

          <section className="results-columns">
            <article className="panel">
              <h3 className="panel-title">Front Analysis</h3>
              <ResultCard
                title="Front"
                score={results.front_analysis?.psl}
                interpretation={results.front_analysis?.interpretation}
                breakdown={results.front_analysis?.breakdown}
              />
              <OverlayViewer
                title="Front Overlay"
                originalSrc={frontPreview}
                overlayData={results.overlays?.front}
              />
            </article>

            <article className="panel">
              <h3 className="panel-title">Side Profile</h3>
              <ResultCard
                title="Side"
                score={results.side_analysis?.psl}
                interpretation={results.side_analysis?.interpretation}
                breakdown={results.side_analysis?.breakdown}
              />
              <OverlayViewer
                title="Side Overlay"
                originalSrc={sidePreview}
                overlayData={results.overlays?.side}
              />
            </article>
          </section>

          <section className="results-columns">
            <article className="panel">
              <MetricsDisplay
                title="Front Measurements"
                measurements={results.front_analysis?.measurements}
                isFront={true}
              />
            </article>
            <article className="panel">
              <MetricsDisplay
                title="Side Measurements"
                measurements={results.side_analysis?.measurements}
                isFront={false}
              />
            </article>
          </section>
        </main>
      )}

      {/* PART B4: Educational-use disclaimer in footer. */}
      <footer className="footer-disclaimer">
        Educational tool only. Scores are algorithmic estimates based on facial landmark proportions and do not define attractiveness.
      </footer>
    </div>
  );
}

export default App;
