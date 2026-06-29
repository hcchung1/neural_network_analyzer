import { useState, useEffect, useCallback, useRef } from 'react'
import { useWebSocket, WSMessage } from './utils/websocket'
import TokenSelector from './components/TokenSelector'
import LayerSlider from './components/LayerSlider'
import InputFeatures from './components/InputFeatures'
import EmbeddingView from './components/EmbeddingView'
import AttentionHeatmap from './components/AttentionHeatmap'
import OutputView from './components/OutputView'

const WS_URL = (() => {
  // 出战
  const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://'
  return `${protocol}${window.location.host}/ws/visualize`
})()

function App() {
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
  const [modelPath, setModelPath] = useState<string>('')

  // Keep refs for latest state in callbacks
  const sendRef = useRef<((msg: WSMessage) => void) | null>(null)
  const requestFeaturesRef = useRef<(() => void) | null>(null)
  const selectedTokenRef = useRef(selectedToken)
  const currentLayerRef = useRef(currentLayer)

  useEffect(() => {
    selectedTokenRef.current = selectedToken
  }, [selectedToken])

  useEffect(() => {
    currentLayerRef.current = currentLayer
  }, [currentLayer])

  // Handle incoming WebSocket messages via callback to avoid React batching issues
  const handleMessage: (rawMsg: WSMessage) => void = useCallback((rawMsg) => {
    const msg = rawMsg as Record<string, unknown>
    const action = msg.action as string

    console.log(`[App] Received ${action} message:`, JSON.stringify(msg, null, 2))

    if (msg.error) {
      setError(msg.error as string)
      setLoading(false)
      return
    }

    switch (action) {
      case 'load_model':
        console.log('[App] load_model response:', JSON.stringify(msg))
        if (msg.error) {
          setError(msg.error as string)
        } else {
          const isLoaded = !!msg.result || msg.result === 'ok' || msg.status === 'ok'
          console.log(`[App] Setting modelLoaded=${isLoaded}, result=${JSON.stringify(msg.result)}`)
          setModelLoaded(isLoaded)
          if (isLoaded) {
            console.log('[App] Model loaded, auto-registering hooks...')
            setTimeout(() => {
              if (sendRef.current) {
                sendRef.current({ action: 'register_hooks', layers: ['embedding', 'attention', 'ffn', 'output'] })
              }
            }, 100)
          }
        }
        setLoading(false)
        break
      case 'register_hooks':
        console.log('[App] register_hooks response:', JSON.stringify(msg))
        if (msg.status === 'ok' || msg.result === 'ok' || msg.result === true) {
          console.log('[App] Hooks registered successfully, requesting features...')
          if (requestFeaturesRef.current) {
            requestFeaturesRef.current()
          }
        } else {
          console.warn('[App] Hooks registration failed:', msg)
        }
        break
      case 'get_token_features':
        setLoading(false)
        console.log('[App] get_token_features data:', msg.data)
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
        console.log('[App] get_token_embedding data:', msg.data)
        if (msg.data && (msg.data as Record<string, unknown>).embedding) {
          setEmbedding(((msg.data as Record<string, unknown>).embedding as number[]))
        }
        break
      case 'get_attention':
        setLoading(false)
        console.log('[App] get_attention data:', msg.data)
        if (msg.data && (msg.data as Record<string, unknown>).attention) {
          setAttention(((msg.data as Record<string, unknown>).attention as number[][]))
        }
        break
      case 'get_output':
        setLoading(false)
        console.log('[App] get_output data:', msg.data)
        if (msg.data) {
          const data = msg.data as Record<string, unknown>
          // Backend returns logits as 2D array [[val1, val2]], extract first element
          if (data.logits && Array.isArray(data.logits) && (data.logits as unknown[]).length > 0) {
            const logitsArray = data.logits as number[][]
            if (logitsArray.length > 0 && Array.isArray(logitsArray[0])) {
              setOutput(logitsArray[0])
            } else {
              setOutput(data.logits as number[])
            }
          }
        }
        break
      case 'run_forward':
        setLoading(false)
        console.log('[App] run_forward result:', msg.result)
        if (requestFeaturesRef.current) {
          requestFeaturesRef.current()
        }
        break
    }
  }, [])

  const { status, send } = useWebSocket(WS_URL, handleMessage)

  // Store send in ref for callbacks
  useEffect(() => {
    sendRef.current = send
  }, [send])

  const requestFeatures = useCallback(() => {
    if (!sendRef.current) return
    setLoading(true)
    setError(null)
    sendRef.current({ action: 'get_token_features', batch_idx: 0, token_idx: selectedTokenRef.current, layer_idx: currentLayerRef.current })
    sendRef.current({ action: 'get_token_embedding', batch_idx: 0, token_idx: selectedTokenRef.current })
    sendRef.current({ action: 'get_attention', layer_idx: currentLayerRef.current })
    sendRef.current({ action: 'get_output' })
  }, [])

  // Store requestFeatures in ref for handleMessage callback
  useEffect(() => {
    requestFeaturesRef.current = requestFeatures
  }, [requestFeatures])

  // Auto-request on token/layer change (only after initial request)
  useEffect(() => {
    if (modelLoaded) {
      requestFeatures()
    }
  }, [selectedToken, currentLayer, modelLoaded, requestFeatures])

  const handleLoadModel = () => {
    setLoading(true)
    setError(null)
    send({ action: 'load_model', path: modelPath || null })
  }

  const handleRegisterHooks = () => {
    send({ action: 'register_hooks', layers: ['embedding', 'attention', 'ffn', 'output'] })
  }

  const handleRunForward = () => {
    // Proper input shape: batch_size=1, seq_len=49, feature_dim=12308
    setLoading(true)
    setError(null)
    const dummyInput = Array.from({ length: 1 }, () =>
      Array.from({ length: 49 }, () =>
        Array.from({ length: 12308 }, () => Math.random() * 2 - 1)
      )
    )
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
            ⏳攻速加成……
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

        {/* Model Path Input */}
        <div style={{ marginBottom: '16px' }}>
          <label
            htmlFor="model-path"
            style={{
              display: 'block',
              marginBottom: '6px',
              fontSize: '14px',
              fontWeight: 500,
              color: '#495057',
            }}
          >
            Model Path
          </label>
          <input
            id="model-path"
            type="text"
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
            placeholder="e.g. /absolute/path/to/model.pt"
            style={{
              width: '100%',
              padding: '8px',
              fontSize: '13px',
              border: '1px solid #ced4da',
              borderRadius: '4px',
              boxSizing: 'border-box',
              fontFamily: 'monospace',
            }}
          />
          <div style={{ marginTop: '4px', fontSize: '12px', color: '#868e96' }}>
            留空：建立一個 dummy Transformer model 進行測試
          </div>
        </div>

        <div style={{ marginBottom: '24px' }}>
          <button
            onClick={handleLoadModel}
            style={{
              width: '100%',
              padding: '10px',
              marginBottom: '8px',
              cursor: 'pointer',
              backgroundColor: '#4dabf7',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
            }}
          >
            Load Model
          </button>
          <button
            onClick={handleRegisterHooks}
            style={{
              width: '100%',
              padding: '10px',
              marginBottom: '8px',
              cursor: 'pointer',
              backgroundColor: '#69db7c',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
            }}
          >
            Register Hooks
          </button>
          <button
            onClick={handleRunForward}
            style={{
              width: '100%',
              padding: '10px',
              cursor: 'pointer',
              backgroundColor: '#ffa94d',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
            }}
          >
            Run Forward
          </button>
        </div>

        {modelLoaded && (
          <div style={{ color: '#28a745', fontSize: '14px', marginBottom: '16px' }}>
            ✅ Model loaded and hooks registered
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
