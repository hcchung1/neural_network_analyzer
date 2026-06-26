import React from 'react'

interface AttentionHeatmapProps {
  attention: number[][] | null
  tokenIndex: number
}

const AttentionHeatmap: React.FC<AttentionHeatmapProps> = ({ attention, tokenIndex }) => {
  if (!attention) {
    return <div style={{ padding: '16px', color: '#888' }}>No attention data</div>
  }

  // attention shape: [seq_len, seq_len] or [heads, seq_len, seq_len]
  let data = attention
  if (Array.isArray(attention[0]) && Array.isArray(attention[0][0])) {
    // Multi-head: take average across heads for display
    const heads = attention as number[][][]
    const seqLen = heads[0].length
    data = Array.from({ length: seqLen }, (_, i) =>
      Array.from({ length: seqLen }, (_, j) =>
        heads.reduce((sum, h) => sum + (h[i]?.[j] || 0), 0) / heads.length
      )
    )
  }

  const flat = data.flat()
  const minVal = Math.min(...flat)
  const maxVal = Math.max(...flat)
  const range = maxVal - minVal || 1

  return (
    <div style={{ padding: '16px' }}>
      <h3>Attention Heatmap (Token {tokenIndex})</h3>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${data.length}, 1fr)`,
          gap: '1px',
          marginTop: '12px',
        }}
      >
        {data.map((row, i) =>
          row.map((val, j) => {
            const normalized = (val - minVal) / range
            return (
              <div
                key={`${i}-${j}`}
                title={`Token ${i} → ${j}: ${val.toFixed(4)}`}
                style={{
                  aspectRatio: '1',
                  backgroundColor: `rgba(255, 100, 50, ${normalized})`,
                  border: i === tokenIndex || j === tokenIndex ? '2px solid #333' : 'none',
                }}
              />
            )
          })
        )}
      </div>
    </div>
  )
}

export default AttentionHeatmap
