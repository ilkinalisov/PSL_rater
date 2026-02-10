import React from 'react';
import '../styles/ResultsCard.css';

const ResultsCard = ({ title, pslScore, interpretation, breakdown }) => {
  if (pslScore === undefined || pslScore === null) {
    return (
      <div className="results-card empty">
        <p>No results to display</p>
      </div>
    );
  }

  const getScoreColor = (score) => {
    if (score >= 9) return '#10B981';
    if (score >= 8) return '#3B82F6';
    if (score >= 7) return '#8B5CF6';
    if (score >= 6) return '#F59E0B';
    if (score >= 5) return '#EC4899';
    if (score >= 4) return '#EF4444';
    return '#DC2626';
  };

  const color = getScoreColor(pslScore);

  return (
    <div className="results-card">
      {title && <h4 className="results-card-title">{title}</h4>}

      <div className="score-card" style={{
        background: `linear-gradient(135deg, ${color}15, ${color}30)`,
        borderColor: color
      }}>
        <div className="score-main">
          <span className="score-number" style={{ color }}>{pslScore}</span>
          <span className="score-out-of">/ 10</span>
        </div>
        <div className="score-label" style={{ color }}>
          {interpretation}
        </div>
      </div>

      {breakdown && (
        <div className="breakdown-section">
          <h5>Component Scores</h5>
          <div className="breakdown-grid">
            {Object.entries(breakdown).map(([key, value]) => {
              const numValue = typeof value === 'number' ? value : parseFloat(value);
              if (isNaN(numValue)) return null;
              const barColor = getScoreColor(numValue);
              return (
                <div key={key} className="breakdown-item">
                  <span className="breakdown-label">
                    {key.replace(/_/g, ' ')}
                  </span>
                  <div className="breakdown-bar">
                    <div
                      className="breakdown-fill"
                      style={{
                        width: `${Math.min(100, numValue * 10)}%`,
                        background: `linear-gradient(90deg, ${barColor}, ${barColor}CC)`
                      }}
                    />
                  </div>
                  <span className="breakdown-value">{numValue.toFixed(1)}/10</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultsCard;
