import React from 'react'

interface OutputViewProps {
  logits: number[] | null
  labels?: string[]
}

const OutputView: React.FC<OutputViewProps> = ({ logits, labels }) => {
  if (!logits) {
    return <div style={{ padding: '16px', color: '#888' }}>No output data</div>
  }

  // Apply softmax
  const maxLogit = Math.max(...logits)
  const expLogits = logits.map((l) => Math.exp(l - maxLogit))
  const sumExp = expLogits.reduce((a, b) => a + b, 0)
  const probs = expLogits.map((e) => e / sumExp)
  const maxProb = Math.max(...probs)

  return (
    <div style={{ padding: '16px' }}>
      <h3>Output Probabilities</h3>
      <div style={{ marginTop: '12px' }}>
        {probs.map((prob, idx) => (
          <div
            key={idx}
            style={{
              display: 'flex',
              alignItems: 'center',
              marginBottom: '8px',
            }}
          >
            <span style={{ width: '80px', fontSize: '12px' }}>
              {labels?.[idx] || `Token ${idx}`}
            </span>
            <div
              style={{
                height: '20px',
                width: `${(prob / (maxProb || 1)) * 300}px`,
                backgroundColor: prob === maxProb ? '#ff6b6b' : '#4dabf7',
                borderRadius: '4px',
                marginRight: '8px',
                transition: 'width 0.3s ease',
              }}
            />
            <span style={{ fontSize: '12px' }}>{(prob * 100).toFixed(2)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default OutputView
