import React from 'react'

interface TokenSelectorProps {
  tokenCount: number
  selectedToken: number
  onSelect: (index: number) => void
}

const TokenSelector: React.FC<TokenSelectorProps> = ({ tokenCount, selectedToken, onSelect }) => {
  return (
    <div style={{ marginBottom: '16px' }}>
      <label style={{ display: 'block', marginBottom: '8px', fontWeight: 600 }}>Token Index</label>
      <select
        value={selectedToken}
        onChange={(e) => onSelect(Number(e.target.value))}
        style={{ padding: '8px', borderRadius: '4px', minWidth: '120px' }}
      >
        {Array.from({ length: tokenCount }, (_, i) => (
          <option key={i} value={i}>
            Token {i}
          </option>
        ))}
      </select>
    </div>
  )
}

export default TokenSelector
