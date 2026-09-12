import React from 'react'

interface LayerInputViewProps {
  vector: number[] | null
  layerIndex: number
  tokenIndex: number
  modelDimension: number | null
}

const LayerInputView: React.FC<LayerInputViewProps> = ({ vector, layerIndex, tokenIndex, modelDimension }) => {
  if (!vector) {
    return <div style={{ padding: '16px', color: '#888' }}>No layer input vector available</div>
  }

  return (
    <div style={{ padding: '16px' }}>
      <h3 style={{ marginBottom: 4 }}>Layer Input Vector</h3>
      <small style={{ color: '#6c757d' }}>
        Token {tokenIndex} immediately before Encoder Block {layerIndex} · d_model={modelDimension ?? vector.length}
      </small>
      <div className="feature-grid">
        {vector.map((val, idx) => {
          const displayVal = Math.abs(val) < 0.01 ? val.toExponential(1) : val.toFixed(2)
          const intensity = Math.min(Math.log10(Math.abs(val) + 1.01) / Math.log10(2), 1)
          return <div
            key={idx}
            className="feature-cell"
            tabIndex={0}
            data-tooltip={`Dimension ${idx} · ${val.toPrecision(10)}`}
            aria-label={`Layer ${layerIndex} input dimension ${idx}: ${val.toPrecision(10)}`}
            style={{
              backgroundColor: val >= 0
                ? `rgba(112, 72, 232, ${0.15 + intensity * 0.85})`
                : `rgba(240, 101, 149, ${0.15 + intensity * 0.85})`,
              color: intensity > 0.5 ? '#fff' : '#333',
            }}
          >{displayVal}</div>
        })}
      </div>
    </div>
  )
}

export default LayerInputView
