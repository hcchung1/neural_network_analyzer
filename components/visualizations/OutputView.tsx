import React from 'react'

export interface ModelOutputSchema {
  output_type: 'binary_logit' | 'multiclass_logits' | 'probability'
  class_names: string[]
  positive_class_index: number | null
  decision_threshold: number
}

interface OutputViewProps {
  logits: number[] | null
  schema: ModelOutputSchema | null
}

function sigmoid(value: number): number {
  return value >= 0
    ? 1 / (1 + Math.exp(-value))
    : Math.exp(value) / (1 + Math.exp(value))
}

function probabilities(logits: number[], schema: ModelOutputSchema): { values: number[]; labels: string[] } | null {
  if (schema.output_type === 'binary_logit') {
    if (logits.length !== 1 || schema.positive_class_index !== 1) return null
    const positive = sigmoid(logits[0])
    return { values: [1 - positive, positive], labels: ['not_tenpai', 'tenpai'] }
  }
  if (schema.output_type === 'probability') {
    if (logits.some(value => value < 0 || value > 1)) return null
    if (logits.length === 1) {
      const positive = logits[0]
      return { values: [1 - positive, positive], labels: ['not_tenpai', 'tenpai'] }
    }
    return { values: logits, labels: schema.class_names }
  }
  const maxLogit = Math.max(...logits)
  const expLogits = logits.map(value => Math.exp(value - maxLogit))
  const total = expLogits.reduce((sum, value) => sum + value, 0)
  return { values: expLogits.map(value => value / total), labels: schema.class_names }
}

const OutputView: React.FC<OutputViewProps> = ({ logits, schema }) => {
  if (!logits) {
    return <div style={{ padding: '16px', color: '#888' }}>No output data</div>
  }

  const resolved = schema ? probabilities(logits, schema) : null
  if (!resolved) {
    return <div style={{ padding: '16px' }}>
      <h3>Raw Model Output</h3>
      <div style={{ color: '#c92a2a', fontSize: 13, marginBottom: 8 }}>
        Output schema unavailable or incompatible; probability conversion was intentionally skipped.
      </div>
      <code>{JSON.stringify(logits)}</code>
    </div>
  }

  const maxProb = Math.max(...resolved.values)
  return (
    <div style={{ padding: '16px' }}>
      <h3>Output Probabilities</h3>
      <small style={{ color: '#6c757d' }}>
        {schema?.output_type} · threshold {schema?.decision_threshold.toFixed(2)}
      </small>
      <div style={{ marginTop: '12px' }}>
        {resolved.values.map((probability, index) => (
          <div key={index} style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ width: '100px', fontSize: '12px' }}>
              {resolved.labels[index] || `Class ${index}`}
            </span>
            <div style={{
              height: '20px',
              width: `${(probability / (maxProb || 1)) * 300}px`,
              backgroundColor: probability === maxProb ? '#ff6b6b' : '#4dabf7',
              borderRadius: '4px',
              marginRight: '8px',
              transition: 'width 0.3s ease',
            }} />
            <span style={{ fontSize: '12px' }}>{(probability * 100).toFixed(2)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default OutputView
