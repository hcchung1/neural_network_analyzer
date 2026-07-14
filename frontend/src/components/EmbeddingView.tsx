import React from 'react'

interface EmbeddingViewProps {
  embedding: number[] | null
  projected?: { x: number; y: number }[] | null
  tokenIndex?: number
}

const EmbeddingView: React.FC<EmbeddingViewProps> = ({ embedding, projected, tokenIndex = 0 }) => {
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
      <h3 style={{ marginBottom: 4 }}>Embedding Vector</h3>
      <small style={{ color: '#6c757d' }}>
        Token {tokenIndex} · input projection into d_model, before positional/meta encoding and Encoder Block 0
      </small>
      <div
        className="embedding-chart"
      >
        {embedding.map((val, idx) => (
          <div
            key={idx}
            className="embedding-bar"
            tabIndex={0}
            data-tooltip={`Dimension ${idx} · ${val.toPrecision(10)}`}
            aria-label={`Embedding dimension ${idx}: ${val.toPrecision(10)}`}
            style={{
              height: `${(Math.abs(val) / (maxVal || 1)) * 100}%`,
              backgroundColor:
                val > 0 ? '#4dabf7' : '#ffa94d',
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
                r={3}
                fill={'#4dabf7'}
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
