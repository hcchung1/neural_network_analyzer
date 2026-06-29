import React from 'react'

interface EmbeddingViewProps {
  embedding: number[] | null
  projected?: { x: number; y: number }[] | null
  highlightIndex?: number
}

const EmbeddingView: React.FC<EmbeddingViewProps> = ({ embedding, projected, highlightIndex = 0 }) => {
  if (!embedding) {
    return <div style={{ padding: '16px', color: '#888' }}>No embedding data</div>
  }

  // Handle non-array embedding values (e.g., scalar returned from backend error)
  if (!Array.isArray(embedding)) {
    return <div style={{ padding: '16px', color: '#e03131' }}>Invalid embedding data: expected array, got {typeof embedding}</div>
  }

  // Simple bar chart visualization of embedding vector
  const maxVal = Math.max(...embedding.map(Math.abs))

  return (
    <div style={{ padding: '16px' }}>
      <h3>Embedding Vector</h3>
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-end',
          height: '200px',
          gap: '1px',
          marginTop: '12px',
          overflowX: 'auto',
        }}
      >
        {embedding.map((val, idx) => (
          <div
            key={idx}
            title={`Dim ${idx}: ${val.toFixed(4)}`}
            style={{
              flex: '1 0 4px',
              height: `${(Math.abs(val) / (maxVal || 1)) * 100}%`,
              backgroundColor:
                idx === highlightIndex ? '#ff6b6b' : val > 0 ? '#4dabf7' : '#ffa94d',
              minHeight: '2px',
            }}
          />
        ))}
      </div>
      {projected && (
        <div style={{ marginTop: '16px' }}>
          <h4>PCA Projection</h4>
          <svg viewBox="0 0 400 300" style={{ width: '100%', height: '300px', border: '1px solid #ddd' }}>
            {projected.map((point, idx) => (
              <circle
                key={idx}
                cx={((point.x + 1) / 2) * 400}
                cy={300 - ((point.y + 1) / 2) * 300}
                r={idx === highlightIndex ? 6 : 3}
                fill={idx === highlightIndex ? '#ff6b6b' : '#4dabf7'}
                opacity={0.7}
              />
            ))}
          </svg>
        </div>
      )}
    </div>
  )
}

export default EmbeddingView
