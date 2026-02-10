import React from 'react';

const DESCRIPTION_MAP = {
  gonial_angle: 'Jaw angle definition',
  nasolabial: 'Nose-lip harmony',
  facial_convexity: 'Profile angle',
  vertical_balance: 'Facial thirds',
  profile_harmony: 'Feature consistency',
  forward_growth: 'Midface projection',
  symmetry: 'Bilateral balance',
  proportions: 'Width-height balance',
  eyes: 'Eye shape',
  jawline: 'Jaw definition',
  harmony: 'Feature consistency',
  golden_ratio: 'Phi proportions',
};

const LABEL_MAP = {
  gonial_angle: 'Gonial Angle',
  nasolabial: 'Nasolabial',
  facial_convexity: 'Facial Convexity',
  vertical_balance: 'Vertical Balance',
  profile_harmony: 'Profile Harmony',
  forward_growth: 'Forward Growth',
  symmetry: 'Symmetry',
  proportions: 'Proportions',
  eyes: 'Canthal Tilt',
  jawline: 'Jawline',
  harmony: 'Harmony',
  golden_ratio: 'Golden Ratio',
};

const getScoreColor = (score) => {
  if (score >= 8.0) return '#fbbf24';
  if (score >= 6.5) return '#3b82f6';
  if (score >= 5.0) return '#e0e0e0';
  if (score >= 3.5) return '#f97316';
  return '#ef4444';
};

const formatLabel = (key) => LABEL_MAP[key] || key.replace(/_/g, ' ').replace(/\b\w/g, (m) => m.toUpperCase());

const ResultCard = ({ title, score, interpretation, breakdown }) => {
  const scoreColor = getScoreColor(Number(score) || 0);

  const rows = Object.entries(breakdown || {})
    .filter(([key]) => !(title === 'Front' && key === 'jawline'))
    .filter(([, value]) => typeof value === 'number' && Number.isFinite(value))
    .map(([key, value]) => ({
      key,
      value,
      label: formatLabel(key),
      description: DESCRIPTION_MAP[key] || 'Facial feature metric',
      color: getScoreColor(value),
    }));

  return (
    <section className="result-card">
      <div className="result-card-header">
        <h3>{title}</h3>
        <div className="result-card-main-score" style={{ color: scoreColor }}>
          {Number(score || 0).toFixed(1)} / 10
        </div>
      </div>
      <p className="result-card-interpretation" style={{ color: scoreColor }}>{interpretation}</p>

      <div className="component-score-grid">
        {rows.map((row) => (
          <article key={row.key} className="component-score-item" style={{ borderLeftColor: row.color }}>
            <div className="component-score-line">
              {row.label}: <strong>{row.value.toFixed(1)}/10</strong> - {row.description}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
};

export default ResultCard;
