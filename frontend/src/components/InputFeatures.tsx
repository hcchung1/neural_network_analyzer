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
        {features.map((val, idx) => (
          <div
            key={idx}
            title={`Index ${idx}: ${val.toFixed(4)}`}
            style={{
              height: '32px',
              backgroundColor: `rgba(70, 130, 180, ${Math.min(Math.abs(val) + 0.1, 1)})`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '10px',
              color: '#fff',
              borderRadius: '2px',
            }}
          >
            {idx}
          </div>
        ))}
      </div>
    </div>
  )
}

export default InputFeatures
