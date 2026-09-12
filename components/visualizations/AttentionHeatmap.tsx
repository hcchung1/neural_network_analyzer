import React from 'react'

interface AttentionHeatmapProps {
  attention: number[][] | number[][][] | null
  tokenIndex: number
}

const AttentionHeatmap: React.FC<AttentionHeatmapProps> = ({ attention, tokenIndex }) => {
  if (!attention) {
    return <div style={{ padding: '16px', color: '#888' }}>No attention data</div>
  }

  // attention shape: [seq_len, seq_len] or [heads, seq_len, seq_len]
  // The data from backend is: [heads, seq_len, seq_len]
  let data: number[][]
  if (attention.length > 0 && Array.isArray(attention[0]) && Array.isArray((attention as number[][][])[0][0])) {
    // Multi-head: take average across heads for display
    const heads = attention as number[][][]
    const seqLen = heads[0].length
    data = Array.from({ length: seqLen }, (_, i) =>
      Array.from({ length: seqLen }, (_, j) =>
        heads.reduce((sum, h) => sum + (h[i][j] || 0), 0) / heads.length
      )
    )
  } else {
    data = attention as number[][]
  }

  // Normalize data: ensure all rows have the same length
  const maxLen = Math.max(...data.map(row => row.length))
  const paddedData = data.map(row => {
    const padded = [...row]
    while (padded.length < maxLen) {
      padded.push(0)  // Pad with zeros
    }
    return padded
  })

  const flat = paddedData.flat()
  const minVal = Math.min(...flat)
  const maxVal = Math.max(...flat)
  const range = maxVal - minVal || 1

  // Check if all values are zero or very small
  const allZero = flat.every(v => Math.abs(v) < 1e-10)
  if (allZero) {
    return (
      <div style={{ padding: '16px' }}>
        <h3>Attention Heatmap (Token {tokenIndex})</h3>
        <div style={{ marginTop: '12px', color: '#888', fontSize: '14px' }}>
          No attention data available.
          <br />
          <strong>Please restart the backend server</strong> to apply the latest changes.
          <br />
          <code style={{ backgroundColor: '#f5f5f5', padding: '2px 4px', borderRadius: '3px' }}>
            cd /workspace/Study/Mahjong-Hidden-Information-Forecast/ArchAnalyzer/backend && python main.py
          </code>
        </div>
      </div>
    )
  }

  return (
    <div style={{ padding: '16px' }}>
      <h3>Attention Heatmap (Token {tokenIndex})</h3>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${paddedData.length}, 1fr)`,
          gap: '1px',
          marginTop: '12px',
        }}
      >
        {paddedData.map((row, i) =>
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
