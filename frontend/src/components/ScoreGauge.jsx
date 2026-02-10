import React from 'react';

const getScoreColor = (score) => {
  if (score >= 8.0) return '#fbbf24';
  if (score >= 6.5) return '#3b82f6';
  if (score >= 5.0) return '#e0e0e0';
  if (score >= 3.5) return '#f97316';
  return '#ef4444';
};

const ScoreGauge = ({ score = 0, label = 'PSL Score' }) => {
  const normalized = Math.max(0, Math.min(10, Number(score) || 0));
  const percent = normalized / 10;
  const radius = 78;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - percent);
  const color = getScoreColor(normalized);

  return (
    <div className="score-gauge" aria-label={`${label} ${normalized.toFixed(1)} out of 10`}>
      <svg width="210" height="210" viewBox="0 0 210 210" role="img">
        <circle cx="105" cy="105" r={radius} className="score-gauge-track" />
        <circle
          cx="105"
          cy="105"
          r={radius}
          className="score-gauge-progress"
          style={{ stroke: color, strokeDasharray: circumference, strokeDashoffset: offset }}
        />
      </svg>
      <div className="score-gauge-center">
        <div className="score-gauge-value" style={{ color }}>{normalized.toFixed(1)}</div>
        <div className="score-gauge-scale">/10</div>
      </div>
    </div>
  );
};

export default ScoreGauge;
