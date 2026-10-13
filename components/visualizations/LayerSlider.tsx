import React from 'react'

interface LayerSliderProps {
  maxLayer: number
  currentLayer: number
  onChange: (layer: number) => void
}

const LayerSlider: React.FC<LayerSliderProps> = ({ maxLayer, currentLayer, onChange }) => {
  return (
    <div style={{ marginBottom: '16px' }}>
      <label style={{ display: 'block', marginBottom: '8px', fontWeight: 600 }}>
        Encoder Block: {currentLayer} <small style={{ color: '#64748b', fontWeight: 400 }}>(0–{maxLayer})</small>
      </label>
      <input
        type="range"
        min={0}
        max={maxLayer}
        value={currentLayer}
        aria-label={`Encoder block 0 through ${maxLayer}`}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-blue-600"
      />
    </div>
  )
}

export default LayerSlider
