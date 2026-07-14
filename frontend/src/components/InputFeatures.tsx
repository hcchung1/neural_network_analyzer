import React from 'react'

interface InputFeaturesProps {
  features: number[] | null
  tokenIndex: number
}

const InputFeatures: React.FC<InputFeaturesProps> = ({ features, tokenIndex }) => {
  if (!features) {
    return <div style={{ padding: '16px', color: '#888' }}>No input features available</div>
  }

  return (
    <div style={{ padding: '16px' }}>
      <details className="vector-details" open>
        <summary>
          <span>Input Features</span>
          <small>Token {tokenIndex} · {features.length.toLocaleString()} raw values</small>
        </summary>
        <div className="feature-grid">
          {features.map((val, idx) => {
          // Format small values with scientific notation
          const displayVal = Math.abs(val) < 0.01 ? val.toExponential(1) : val.toFixed(2)
          // Calculate color intensity based on absolute value (log scale for better visibility)
          const intensity = Math.min(Math.log10(Math.abs(val) + 1.01) / Math.log10(2), 1)
          const isPositive = val >= 0
          return (
            <div
              key={idx}
              className="feature-cell"
              tabIndex={0}
              data-tooltip={`Feature ${idx} · ${val.toPrecision(10)}`}
              aria-label={`Feature ${idx}: ${val.toPrecision(10)}`}
              style={{
                backgroundColor: isPositive
                  ? `rgba(70, 130, 180, ${0.15 + intensity * 0.85})`
                  : `rgba(220, 80, 80, ${0.15 + intensity * 0.85})`,
                color: intensity > 0.5 ? '#fff' : '#333',
              }}
            >
              {displayVal}
            </div>
          )
          })}
        </div>
      </details>
    </div>
  )
}

export default InputFeatures
