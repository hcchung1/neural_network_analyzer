import React, { useState, useEffect, useCallback, useRef } from 'react'
import { useWebSocket } from './utils/websocket'
import TokenSelector from './components/TokenSelector'
import LayerSlider from './components/LayerSlider'
import InputFeatures from './components/InputFeatures'
import EmbeddingView from './components/EmbeddingView'
import AttentionHeatmap from './components/AttentionHeatmap'
import OutputView from './components/OutputView'

const WS_URL = 'ws://localhost:8000/ws/visualize'

function App() {
  const { status, lastMessage, send } = useWebSocket(WS_URL)
  const [selectedToken, setSelectedToken] = useState(0)
  const [currentLayer, setCurrentLayer] = useState(0)
  const [maxLayer] = useState(4)
  const [tokenCount] = useState(16)

  // Data states
  const [inputFeatures, setInputFeatures] = useState<number[] | null>(null)
  const [embedding, setEmbedding] = useState<number[] | null>(null)
  const [attention, setAttention] = useState<number[][] | null>(null)
  const [output, setOutput] = useState<number[] | null>(null)
  const [modelLoaded, setModelLoaded] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const hasRequested = useRef(false)

  // Handle incoming WebSocket messages
  useEffect(() => {
    if (!lastMessage || typeof lastMessage !== 'object') return

    const msg = lastMessage as Record<string, unknown>
    const action = msg.action as string

    if (msg.error) {
      setError(msg.error as string)
      setLoading(false)
      return
    }

    switch (action) {
      case 'load_model':
        setModelLoaded(!!msg.result)
        setLoading(false)
        break
      case 'get_token_features':
        setLoading(false)
        if (msg.data && typeof msg.data === 'object') {
          const data = msg.data as Record<string, unknown>
          const firstKey = Object.keys(data)[0]
          if (firstKey && Array.isArray(data[firstKey])) {
            setInputFeatures(data[firstKey] as number[])
          }
        }
        break
      case 'get_token_embedding':
        setLoading(false)
        if (msg.data && (msg.data as Record<string, unknown>).embedding) {
          setEmbedding(((msg.data as Record<string, unknown>).embedding as number[]))
        }
        break
      case 'get_attention':
        setLoading(false)
        if (msg.data && (msg.data as Record<string, unknown>).attention) {
          setAttention(((msg.data as Record<string, unknown>).attention as number[][]))
        }
        break
      case 'get_output':
        setLoading(false)
        if (msg.data && (msg.data as Record<string, unknown>).output) {
          const out = (msg.data as Record<string, unknown>).output
          if (Array.isArray(out)) {
            setOutput(out as number[])
          }
        }
        break
      case 'run_forward':
        setLoading(false)
        // After forward, re-request all features to refresh the UI
        requestFeatures()
        break
    }
  }, [lastMessage])

  const requestFeatures = useCallback(() => {
    if (status !== 'open') return
    setLoading(true)
    setError(null)
    send({ action: 'get_token_features', batch_idx: 0, token_idx: selectedToken, layer_idx: currentLayer })
    send({ action: 'get_token_embedding', batch_idx: 0, token_idx: selectedToken })
    send({ action: 'get_attention', layer_idx: currentLayer })
    send({ action: 'get_output' })
  }, [send, selectedToken, currentLayer, status])

  // Auto-request on token/layer change (only after initial request)
  useEffect(() => {
    if (!hasRequested.current) {
      hasRequested.current = true
      return
    }
    requestFeatures()
  }, [selectedToken, currentLayer, requestFeatures])

  const handleLoadModel = () => {
    setLoading(true)
    setError(null)
    send({ action: 'load_model', path: null })
  }

  const handleRegisterHooks = () => {
    send({ action: 'register_hooks', layers: ['embedding', 'attention', 'ffn', 'output'] })
  }

  const handleRunForward = () => {
    // Dummy input: batch_size=1, seq_len=16
    setLoading(true)
    setError(null)
    const dummyInput = [Array.from({ length: 16 }, (_, i) => i % 128)]
    send({ action: 'run_forward', input: dummyInput })
  }

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: 'system-ui, sans-serif' }}>
      {/* Sidebar */}
      <aside
        style={{
          width: '280px',
          backgroundColor: '#f8f9fa',
          borderRight: '1px solid #dee2e6',
          padding: '16px',
          overflowY: 'auto',
        }}
      >
        <h2 style={{ marginTop: 0, marginBottom: '24px' }}>ArchAnalyzer</h2>

        <div style={{ marginBottom: '16px' }}>
          <span
            style={{
              display: 'inline-block',
              padding: '4px 12px',
              borderRadius: '12px',
              fontSize: '12px',
              backgroundColor:
                status === 'open' ? '#d4edda' : status === 'error' ? '#f8d7da' : '#fff3cd',
              color:
                status === 'open' ? '#155724' : status === 'error' ? '#721c24' : '#856404',
            }}
          >
            WS: {status}
          </span>
        </div>

        {loading && (
          <div style={{ padding: '8px 0', color: '#666', fontSize: '14px' }}>
            ⏳ Loading...
          </div>
        )}

        {error && (
          <div
            style={{
              padding: '12px',
              backgroundColor: '#f8d7da',
              color: '#721c24',
              borderRadius: '4px',
              fontSize: '14px',
              marginBottom: '16px',
            }}
          >
            ❌ {error}
          </div>
        )}

        <div style={{ marginBottom: '24px' }}>
          <button
            onClick={handleLoadModel}
            disabled={loading}
            style={{
              width: '100%',
              padding: '10px',
              marginBottom: '8px',
              cursor: loading ? 'not-allowed' : 'pointer',
              backgroundColor: '#4dabf7',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              opacity: loading ? 0.7 : 1,
            }}
          >
            Load Model
          </button>
          <button
            onClick={handleRegisterHooks}
            disabled={loading}
            style={{
              width: '100%',
              padding: '10px',
              marginBottom: '8px',
              cursor: loading ? 'not-allowed' : 'pointer',
              backgroundColor: '#69db7c',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              opacity: loading ? 0.7 : 1,
            }}
          >
            Register Hooks
          </button>
          <button
            onClick={handleRunForward}
            disabled={loading}
            style={{
              width: '100%',
              padding: '10px',
              cursor: loading ? 'not-allowed' : 'pointer',
              backgroundColor: '#ffa94d',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              opacity: loading ? 0.7 : 1,
            }}
          >
            Run Forward
          </button>
        </div>

        {modelLoaded && (
          <div style={{ color: '#28a745', fontSize: '14px', marginBottom: '16px' }}>
            Model loaded successfully
          </div>
        )}

        <TokenSelector tokenCount={tokenCount} selectedToken={selectedToken} onSelect={setSelectedToken} />
        <LayerSlider maxLayer={maxLayer} currentLayer={currentLayer} onChange={setCurrentLayer} />
      </aside>

      {/* Main content */}
      <main style={{ flex: 1, overflowY: 'auto', padding: '16px' }}>
        <h1 style={{ marginTop: 0, marginBottom: '24px' }}>Transformer Token Visualization</h1>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '16px',
          }}
        >
          <div style={{ border: '1px solid #dee2e6', borderRadius: '8px', overflow: 'hidden' }}>
            <InputFeatures features={inputFeatures} />
          </div>
          <div style={{ border: '1px solid #dee2e6', borderRadius: '8px', overflow: 'hidden' }}>
            <EmbeddingView embedding={embedding} highlightIndex={selectedToken} />
          </div>
          <div style={{ border: '1px solid #dee2e6', borderRadius: '8px', overflow: 'hidden' }}>
            <AttentionHeatmap attention={attention} tokenIndex={selectedToken} />
          </div>
          <div style={{ border: '1px solid #dee2e6', borderRadius: '8px', overflow: 'hidden' }}>
            <OutputView logits={output} />
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
