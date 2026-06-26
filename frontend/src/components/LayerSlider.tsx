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
        Layer: {currentLayer}
      </label>
      <input
        type="range"
        min={0}
        max={maxLayer}
        value={currentLayer}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: '100%' }}
      />
    </div>
  )
}

export default LayerSlider
