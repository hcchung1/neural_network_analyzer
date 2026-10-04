import React, { useEffect, useMemo, useRef, useState } from 'react'
import { attentionCell, attentionMatrix } from '@/lib/performance-ui'

interface AttentionHeatmapProps {
  attention: number[][] | number[][][] | null
  tokenIndex: number
}

const AttentionHeatmap: React.FC<AttentionHeatmapProps> = ({ attention, tokenIndex }) => {
  const matrix = useMemo(() => attentionMatrix(attention), [attention])
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hover, setHover] = useState<{ row: number; column: number } | null>(null)
  useEffect(() => { setHover(null) }, [attention])
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || matrix.allZero) return
    const context = canvas.getContext('2d')
    if (!context) return
    // One raster pixel per cell: no DOM nodes or oversized canvas for long sequences.
    const pixels = context.createImageData(matrix.columns, matrix.rows)
    const range = matrix.max - matrix.min || 1
    for (let row = 0; row < matrix.rows; row++) {
      for (let column = 0; column < matrix.columns; column++) {
        const offset = (row * matrix.columns + column) * 4
        const normalized = (matrix.values[row][column] - matrix.min) / range
        pixels.data[offset] = 255
        pixels.data[offset + 1] = Math.round(255 - 155 * normalized)
        pixels.data[offset + 2] = Math.round(255 - 205 * normalized)
        pixels.data[offset + 3] = 255
      }
    }
    context.putImageData(pixels, 0, 0)
    context.fillStyle = 'rgba(37, 99, 235, 0.28)'
    if (tokenIndex >= 0 && tokenIndex < matrix.rows) context.fillRect(0, tokenIndex, matrix.columns, 1)
    if (tokenIndex >= 0 && tokenIndex < matrix.columns) context.fillRect(tokenIndex, 0, 1, matrix.rows)
  }, [matrix, tokenIndex])

  const hoveredValue = hover ? matrix.values[hover.row]?.[hover.column] : undefined
  if (!attention) return <div style={{ padding: '16px', color: '#888' }}>No attention data</div>
  return <div style={{ padding: '16px' }}>
    <h3>Attention Heatmap (Token {tokenIndex})</h3>
    {matrix.allZero ? <div style={{ marginTop: '12px', color: '#888', fontSize: '14px' }}>
      No attention data available.<br />
      Confirm that the model and sample loaded successfully, then run inference again.
    </div> : <>
      <canvas ref={canvasRef} role="img" aria-label="Attention heatmap"
        data-selected-token={tokenIndex} width={matrix.columns} height={matrix.rows}
        style={{ display: 'block', width: '100%', aspectRatio: `${matrix.columns} / ${matrix.rows}`, marginTop: 12, imageRendering: 'pixelated' }}
        onPointerMove={event => {
          const bounds = event.currentTarget.getBoundingClientRect()
          const cell = attentionCell(event.clientX - bounds.left, event.clientY - bounds.top, bounds.width, bounds.height, matrix.rows, matrix.columns)
          setHover(previous => previous?.row === cell?.row && previous?.column === cell?.column ? previous : cell)
        }} onPointerLeave={() => setHover(null)}>
        Attention values for {matrix.rows} query tokens and {matrix.columns} key tokens.
      </canvas>
      {hover && hoveredValue !== undefined && <div role="tooltip" style={{ marginTop: 8, fontSize: 12 }}>
        Token {hover.row} → {hover.column}: {hoveredValue.toFixed(4)}
      </div>}
    </>}
  </div>
}
export default AttentionHeatmap
