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
        aria-label="Token Index"
        value={selectedToken}
        onChange={(e) => onSelect(Number(e.target.value))}
        className="min-w-32 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
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
