import React from 'react';
import '../styles/main.css';

const MetricsDisplay = ({ title, measurements, isFront = true }) => {
  if (!measurements) return null;

  const frontMetrics = [
    { key: 'fwhr', label: 'fWHR Ratio', format: (v) => v.toFixed(3) },
    { key: 'midface_ratio', label: 'Midface Ratio', format: (v) => v.toFixed(3) },
    { key: 'eye_separation_ratio', label: 'Eye Separation', format: (v) => v.toFixed(3) },
    { key: 'jaw_to_cheek_ratio', label: 'Jaw to Cheek', format: (v) => v.toFixed(3) },
    { key: 'chin_to_philtrum_ratio', label: 'Chin to Philtrum', format: (v) => v.toFixed(3) },
    { key: 'facial_symmetry_score', label: 'Symmetry', format: (v) => `${v.toFixed(1)}%` },
    { key: 'canthal_tilt_avg', label: 'Canthal Tilt', format: (v) => `${v.toFixed(1)}°` },
    { key: 'nasal_index', label: 'Nasal Index', format: (v) => v.toFixed(3) },
  ];

  const sideMetrics = [
    { key: 'facial_convexity_angle', label: 'Facial Convexity', format: (v) => `${v.toFixed(1)}°` },
    { key: 'nasolabial_angle', label: 'Nasolabial Angle', format: (v) => `${v.toFixed(1)}°` },
    { key: 'gonial_angle', label: 'Gonial Angle', format: (v) => `${v.toFixed(1)}°` },
    { key: 'nasofrontal_angle', label: 'Nasofrontal Angle', format: (v) => `${v.toFixed(1)}°` },
    { key: 'profile_harmony_score', label: 'Profile Harmony', format: (v) => `${v.toFixed(1)}%` },
    { key: 'nasal_projection', label: 'Nasal Projection', format: (v) => v.toFixed(3) },
    { key: 'chin_projection', label: 'Chin Projection', format: (v) => v.toFixed(3) },
    { key: 'lip_projection', label: 'Lip Projection', format: (v) => v.toFixed(3) },
  ];

  const metrics = isFront ? frontMetrics : sideMetrics;

  return (
    <div className="metrics-display">
      {title && <h4>{title}</h4>}
      <div className="measurement-table">
        <table>
          <thead>
            <tr>
              <th>Measurement</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map(({ key, label, format }) => {
              const value = measurements[key];
              if (value === undefined || value === null) return null;

              return (
                <tr key={key}>
                  <td>{label}</td>
                  <td>{format(value)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default MetricsDisplay;
