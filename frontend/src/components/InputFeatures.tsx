import React from 'react'

interface InputFeaturesProps {
  features: number[] | null
}

const InputFeatures: React.FC<InputFeaturesProps> = ({ features }) => {
  if (!features) {
    return <div style={{ padding: '16px', color: '#888' }}>No input features available</div>
  }

  return (
    <div style={{ padding: '16px' }}>
      <h3>Input Features</h3>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(40px, 1fr))',
          gap: '4px',
          marginTop: '12px',
        }}
      >
        {features.map((val, idx) => {
          // Format small values with scientific notation
          const displayVal = Math.abs(val) < 0.01 ? val.toExponential(1) : val.toFixed(2)
          // Calculate color intensity based on absolute value (log scale for better visibility)
          const intensity = Math.min(Math.log10(Math.abs(val) + 1.01) / Math.log10(2), 1)
          const isPositive = val >= 0
          return (
            <div
              key={idx}
              title={`Index ${idx}: ${val.toExponential(4)}`}
              style={{
                height: '32px',
                backgroundColor: isPositive
                  ? `rgba(70, 130, 180, ${0.15 + intensity * 0.85})`
                  : `rgba(220, 80, 80, ${0.15 + intensity * 0.85})`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '9px',
                color: intensity > 0.5 ? '#fff' : '#333',
                borderRadius: '2px',
                border: '1px solid #ddd',
              }}
            >
              {displayVal}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default InputFeatures
