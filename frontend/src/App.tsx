import { useState, useEffect, useCallback, useRef } from 'react'
import { useWebSocket, WSMessage } from './utils/websocket'
import TokenSelector from './components/TokenSelector'
import LayerSlider from './components/LayerSlider'
import InputFeatures from './components/InputFeatures'
import LayerInputView from './components/LayerInputView'
import EmbeddingView from './components/EmbeddingView'
import AttentionHeatmap from './components/AttentionHeatmap'
import OutputView, { ModelOutputSchema } from './components/OutputView'
import TrainingResultsPage from './pages/TrainingResultsPage'
import CsvReaderPage from './pages/CsvReaderPage'
import BinarySampleLoader from './components/BinarySampleLoader'
import VisualizationChatSidebar from './components/VisualizationChatSidebar'

const WS_URL = (() => {
  // 出战
  const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://'
  return `${protocol}${window.location.host}/ws/visualize`
})()

interface ModelInputSchema {
  dtype: 'float32'
  rank: 3
  batch_size: 1
  seq_len: number
  feature_dim: number
  shape: [1, number, number]
}

function parseInputSchema(value: unknown): ModelInputSchema | null {
  if (!value || typeof value !== 'object') return null
  const schema = value as Record<string, unknown>
  const seqLen = schema.seq_len
  const featureDim = schema.feature_dim
  if (!Number.isInteger(seqLen) || !Number.isInteger(featureDim)) return null
  if ((seqLen as number) <= 0 || (featureDim as number) <= 0) return null
  return {
    dtype: 'float32',
    rank: 3,
    batch_size: 1,
    seq_len: seqLen as number,
    feature_dim: featureDim as number,
    shape: [1, seqLen as number, featureDim as number]
  }
}

function parseOutputSchema(value: unknown): ModelOutputSchema | null {
  if (!value || typeof value !== 'object') return null
  const schema = value as Record<string, unknown>
  const outputType = schema.output_type
  const positiveClassIndex = schema.positive_class_index
  const threshold = schema.decision_threshold
  if (!['binary_logit', 'multiclass_logits', 'probability'].includes(String(outputType))) return null
  if (positiveClassIndex !== null && !Number.isInteger(positiveClassIndex)) return null
  if (typeof threshold !== 'number' || threshold < 0 || threshold > 1) return null
  return {
    output_type: outputType as ModelOutputSchema['output_type'],
    class_names: Array.isArray(schema.class_names) ? schema.class_names.map(String) : [],
    positive_class_index: positiveClassIndex as number | null,
    decision_threshold: threshold,
  }
}

function parseCustomFeatureJson(text: string, schema: ModelInputSchema): number[][][] {
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch (error) {
    throw new Error(`Invalid JSON: ${error instanceof Error ? error.message : String(error)}`)
  }
  if (!Array.isArray(parsed) || parsed.length === 0) {
    throw new Error('Feature must be a non-empty JSON array.')
  }

  const isThreeDimensional = Array.isArray(parsed[0]) && Array.isArray(parsed[0][0])
  let matrix: unknown
  if (isThreeDimensional) {
    if (parsed.length !== 1) {
      throw new Error(`Expected batch size 1, received ${parsed.length}.`)
    }
    matrix = parsed[0]
  } else {
    matrix = parsed
  }
  if (!Array.isArray(matrix) || matrix.length !== schema.seq_len) {
    const received = Array.isArray(matrix) ? matrix.length : 0
    throw new Error(`Expected ${schema.seq_len} sequence rows, received ${received}.`)
  }

  const validated = matrix.map((row, rowIndex) => {
    if (!Array.isArray(row) || row.length !== schema.feature_dim) {
      const received = Array.isArray(row) ? row.length : 0
      throw new Error(`Row ${rowIndex} must contain ${schema.feature_dim} values; received ${received}.`)
    }
    return row.map((value, columnIndex) => {
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        throw new Error(`Feature [${rowIndex}][${columnIndex}] must be a finite number.`)
      }
      return value
    })
  })
  return [validated]
}

function App() {
  const [selectedToken, setSelectedToken] = useState(0)
  const [currentLayer, setCurrentLayer] = useState(0)
  const [maxLayer, setMaxLayer] = useState(3)
  const [tokenCount, setTokenCount] = useState(16)
  const [currentView, setCurrentView] = useState<'visualize' | 'training' | 'csvReader'>('visualize')
  const [requestedTrainingFolder, setRequestedTrainingFolder] = useState<{ path: string; requestId: number } | null>(null)

  // Data states
  const [inputFeatures, setInputFeatures] = useState<number[] | null>(null)
  const [embedding, setEmbedding] = useState<number[] | null>(null)
  const [layerInput, setLayerInput] = useState<number[] | null>(null)
  const [attention, setAttention] = useState<number[][] | null>(null)
  const [output, setOutput] = useState<number[] | null>(null)
  const [modelLoaded, setModelLoaded] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [modelPath, setModelPath] = useState<string>('')
  const [modelInputSchema, setModelInputSchema] = useState<ModelInputSchema | null>(null)
  const [modelOutputSchema, setModelOutputSchema] = useState<ModelOutputSchema | null>(null)
  const [modelDimension, setModelDimension] = useState<number | null>(null)
  const [customFeatureError, setCustomFeatureError] = useState<string | null>(null)
  const [boardUrl, setBoardUrl] = useState('')
  const [boardOpen, setBoardOpen] = useState(true)
  const [boardWidth, setBoardWidth] = useState(520)
  const [isResizingBoard, setIsResizingBoard] = useState(false)
  const [autoHideNavigation, setAutoHideNavigation] = useState(false)
  const [chatOpen, setChatOpen] = useState(false)
  // Transient metadata from the last binary sample load
  const [sampleMeta, setSampleMeta] = useState<Record<string, unknown> | null>(null)
  // Keep the current feature matrix for chat context
  const featureMatrixRef = useRef<number[][] | null>(null)

  // Keep refs for latest state in callbacks
  const sendRef = useRef<((msg: WSMessage) => void) | null>(null)
  const requestFeaturesRef = useRef<(() => void) | null>(null)
  const selectedTokenRef = useRef(selectedToken)
  const currentLayerRef = useRef(currentLayer)
  const customFeatureTextRef = useRef<HTMLTextAreaElement>(null)
  const boardPaneRef = useRef<HTMLElement>(null)
  const boardSeparatorRef = useRef<HTMLDivElement>(null)
  const boardResizeRafRef = useRef<number | null>(null)
  const boardResizePointerIdRef = useRef<number | null>(null)
  const boardResizePendingXRef = useRef<number | null>(null)
  const boardResizeWidthRef = useRef(boardWidth)

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
      if (action === 'run_forward') {
        setCustomFeatureError(msg.error as string)
      }
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
          // Update token count and max layer from model metadata
          const meta = (msg.model_meta as Record<string, unknown>) || {}
          const seqLen = typeof meta.seq_len === 'number' ? meta.seq_len : 49
          const nLayers = typeof meta.n_layers === 'number' && meta.n_layers > 0 ? meta.n_layers : 4
          const maxLayerIndex = nLayers - 1
          const dModel = typeof meta.d_model === 'number' && meta.d_model > 0 ? meta.d_model : null
          const inputSchema = parseInputSchema(meta.input_schema)
          const outputSchema = parseOutputSchema(meta.output_schema)
          console.log(`[App] Model metadata: seq_len=${seqLen}, n_layers=${nLayers}`)
          setTokenCount(seqLen)
          setMaxLayer(maxLayerIndex)
          setCurrentLayer((layer) => Math.min(layer, maxLayerIndex))
          setModelDimension(dModel)
          setModelInputSchema(inputSchema)
          setModelOutputSchema(outputSchema)
          setCustomFeatureError(inputSchema ? null : 'The loaded model does not expose a dense feature schema.')
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
          console.log('[App] Hooks registered successfully, running forward pass...')
          // Trigger a forward pass to populate attention data and other cached features
          if (sendRef.current) {
            // Let backend auto-generate dummy input with correct dimensions
            sendRef.current({ action: 'run_forward' })
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
          if (typeof data.error === 'string') {
            setInputFeatures(null)
            setError(data.error)
            break
          }
          setInputFeatures(Array.isArray(data.input) ? data.input as number[] : null)
        } else setInputFeatures(null)
        break
      case 'get_token_embedding':
        setLoading(false)
        console.log('[App] get_token_embedding data:', msg.data)
        if (msg.data && Array.isArray((msg.data as Record<string, unknown>).embedding)) {
          setEmbedding((msg.data as Record<string, unknown>).embedding as number[])
        } else setEmbedding(null)
        break
      case 'get_layer_input':
        setLoading(false)
        console.log('[App] get_layer_input data:', msg.data)
        if (msg.data && Array.isArray((msg.data as Record<string, unknown>).vector)) {
          const data = msg.data as Record<string, unknown>
          setLayerInput(data.vector as number[])
          if (typeof data.dimension === 'number') setModelDimension(data.dimension)
        } else setLayerInput(null)
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
          if (typeof data.error === 'string') {
            setOutput(null)
            setError(data.error)
            break
          }
          // Backend returns logits as 2D array [[val1, val2]], extract first element
          if (data.logits && Array.isArray(data.logits) && (data.logits as unknown[]).length > 0) {
            const logitsArray = data.logits as number[][]
            if (logitsArray.length > 0 && Array.isArray(logitsArray[0])) {
              setOutput(logitsArray[0])
            } else {
              setOutput(data.logits as number[])
            }
          } else setOutput(null)
        } else setOutput(null)
        break
      case 'run_forward':
        setLoading(false)
        console.log('[App] run_forward result:', msg.result)
        if (msg.result && typeof msg.result === 'object' && (msg.result as Record<string, unknown>).error) {
          const forwardError = String((msg.result as Record<string, unknown>).error)
          setCustomFeatureError(forwardError)
          setError(forwardError)
          return
        }
        if (requestFeaturesRef.current) {
          requestFeaturesRef.current()
        }
        break
    }
  }, [])

  const isVisualizerView = currentView === 'visualize'
  const { status, send } = useWebSocket(WS_URL, handleMessage, isVisualizerView)

  // Store send in ref for callbacks
  useEffect(() => {
    sendRef.current = send
  }, [send])

  const requestFeatures = useCallback(() => {
    if (!sendRef.current) return
    setLoading(true)
    setError(null)
    sendRef.current({ action: 'get_token_features', batch_idx: 0, token_idx: selectedTokenRef.current })
    sendRef.current({ action: 'get_token_embedding', batch_idx: 0, token_idx: selectedTokenRef.current })
    sendRef.current({ action: 'get_layer_input', batch_idx: 0, token_idx: selectedTokenRef.current, layer_idx: currentLayerRef.current })
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
    setModelInputSchema(null)
    setModelOutputSchema(null)
    setModelDimension(null)
    setInputFeatures(null)
    setEmbedding(null)
    setLayerInput(null)
    setAttention(null)
    setOutput(null)
    setCustomFeatureError(null)
    featureMatrixRef.current = null
    setSampleMeta(null)
    if (customFeatureTextRef.current) customFeatureTextRef.current.value = ''
    send({ action: 'load_model', path: modelPath || null })
  }

  const addCsvModelToVisualizer = useCallback((path: string) => {
    setModelPath(path)
    setCurrentView('visualize')
    setError(null)
  }, [])

  const addCsvTrainingResults = useCallback((path: string) => {
    setRequestedTrainingFolder({ path, requestId: Date.now() })
    setCurrentView('training')
  }, [])

  const handleRegisterHooks = () => {
    send({ action: 'register_hooks', layers: ['embedding', 'attention', 'ffn', 'output'] })
  }

  const handleRunForward = () => {
    setLoading(true)
    setError(null)
    setCustomFeatureError(null)

    const customFeatureText = customFeatureTextRef.current?.value.trim() ?? ''
    if (!customFeatureText) {
      featureMatrixRef.current = null
      setSampleMeta(null)
      send({ action: 'run_forward' })
      return
    }
    if (!modelInputSchema) {
      setLoading(false)
      setCustomFeatureError('Load a model with an inferred input schema before using custom features.')
      return
    }

    try {
      const customInput = parseCustomFeatureJson(customFeatureText, modelInputSchema)
      featureMatrixRef.current = customInput[0]
      setSampleMeta(null)
      send({ action: 'run_forward', input: customInput })
    } catch (validationError) {
      setLoading(false)
      setCustomFeatureError(validationError instanceof Error ? validationError.message : String(validationError))
    }
  }

  const handleBinaryFeature = useCallback((feature: number[][], validTokenCount: number, meta?: Record<string, unknown>) => {
    if (customFeatureTextRef.current) {
      customFeatureTextRef.current.value = JSON.stringify(feature)
    }
    featureMatrixRef.current = feature
    if (meta) setSampleMeta(meta)
    setInputFeatures(null)
    setEmbedding(null)
    setLayerInput(null)
    setAttention(null)
    setOutput(null)
    if (!sendRef.current) {
      setError('WebSocket 未連線，無法 Run Forward。請確認 backend 與 WS 狀態。')
      return
    }
    setSelectedToken(0)
    setTokenCount(Math.max(1, validTokenCount))
    setLoading(true)
    setError(null)
    setCustomFeatureError(null)
    // Batch dim [1, seq, feat] — matches parseCustomFeatureJson / backend engine.forward
    sendRef.current({ action: 'run_forward', input: [feature] })
  }, [])

  const applyPendingBoardResize = useCallback(() => {
    boardResizeRafRef.current = null
    const clientX = boardResizePendingXRef.current
    if (clientX === null) return
    boardResizePendingXRef.current = null
    const width = Math.max(300, Math.min(1200, window.innerWidth - clientX))
    boardResizeWidthRef.current = width
    if (boardPaneRef.current) boardPaneRef.current.style.width = `${width}px`
    if (boardSeparatorRef.current) {
      boardSeparatorRef.current.style.right = `${width - 4}px`
      boardSeparatorRef.current.setAttribute('aria-valuenow', String(Math.round(width)))
    }
  }, [])

  const queueBoardResize = useCallback((clientX: number) => {
    boardResizePendingXRef.current = clientX
    if (boardResizeRafRef.current === null) {
      boardResizeRafRef.current = requestAnimationFrame(applyPendingBoardResize)
    }
  }, [applyPendingBoardResize])

  const finishBoardResize = useCallback((pointerId?: number) => {
    const activePointerId = boardResizePointerIdRef.current
    if (activePointerId === null || (pointerId !== undefined && pointerId !== activePointerId)) return
    if (boardResizeRafRef.current !== null) {
      cancelAnimationFrame(boardResizeRafRef.current)
      boardResizeRafRef.current = null
    }
    applyPendingBoardResize()
    boardResizePointerIdRef.current = null
    const separator = boardSeparatorRef.current
    if (separator?.hasPointerCapture(activePointerId)) separator.releasePointerCapture(activePointerId)
    setIsResizingBoard(false)
    setBoardWidth(boardResizeWidthRef.current)
  }, [applyPendingBoardResize])

  const startBoardResize = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || boardResizePointerIdRef.current !== null) return
    event.preventDefault()
    boardResizePointerIdRef.current = event.pointerId
    boardResizeWidthRef.current = boardWidth
    boardResizePendingXRef.current = event.clientX
    event.currentTarget.setPointerCapture(event.pointerId)
    setIsResizingBoard(true)
  }, [boardWidth])

  const moveBoardResize = useCallback((event: React.PointerEvent) => {
    if (event.pointerId === boardResizePointerIdRef.current) queueBoardResize(event.clientX)
  }, [queueBoardResize])

  const endBoardResize = useCallback((event: React.PointerEvent) => {
    finishBoardResize(event.pointerId)
  }, [finishBoardResize])

  useEffect(() => {
    if (!isResizingBoard) return
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    const onWindowBlur = () => finishBoardResize()
    window.addEventListener('blur', onWindowBlur)
    return () => {
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      window.removeEventListener('blur', onWindowBlur)
    }
  }, [finishBoardResize, isResizingBoard])

  useEffect(() => () => {
    if (boardResizeRafRef.current !== null) cancelAnimationFrame(boardResizeRafRef.current)
  }, [])

  return (
    <div className="app-shell">
      <div className={`app-nav-rail ${autoHideNavigation ? 'auto-hide' : ''}`}>
        <aside className="app-sidebar">
          <header className="app-sidebar-header">
            <div><strong>ArchAnalyzer</strong><small>Model inspection suite</small></div>
            <button
              className="nav-mode-toggle"
              onClick={() => setAutoHideNavigation((value) => !value)}
              title={autoHideNavigation ? '固定側邊欄' : '切換為自動隱藏'}
              aria-label={autoHideNavigation ? 'Pin navigation sidebar' : 'Auto-hide navigation sidebar'}
            >{autoHideNavigation ? '📌' : '◀'}</button>
          </header>
          <nav className="app-navigation" aria-label="Main pages">
            <button className={currentView === 'visualize' ? 'active' : ''} onClick={() => setCurrentView('visualize')}><span>◫</span><div>Token Visualization<small>Transformer internals</small></div></button>
            <button className={currentView === 'training' ? 'active' : ''} onClick={() => setCurrentView('training')}><span>⌁</span><div>Training Results<small>Metrics and comparisons</small></div></button>
            <button className={currentView === 'csvReader' ? 'active' : ''} onClick={() => setCurrentView('csvReader')}><span>▤</span><div>CSV Reader<small>Tenhou result explorer</small></div></button>
          </nav>
          {autoHideNavigation && <div className="nav-hover-hint" aria-hidden="true">›</div>}
        </aside>
      </div>

      {/* Main content */}
      <main style={{ flex: 1, minWidth: 0, overflowY: 'auto', padding: '16px' }}>
        <div style={{ display: currentView === 'training' ? 'block' : 'none' }}>
          <TrainingResultsPage onBack={() => setCurrentView('visualize')} requestedFolder={requestedTrainingFolder} />
        </div>

        <div style={{ display: currentView === 'csvReader' ? 'block' : 'none', height: '100%' }}>
          <CsvReaderPage
            onBack={() => setCurrentView('visualize')}
            onAddModel={addCsvModelToVisualizer}
            onAddTrainingResults={addCsvTrainingResults}
          />
        </div>

        <div style={{ display: currentView === 'visualize' ? 'block' : 'none' }}>
          <h1 style={{ marginTop: 0, marginBottom: '24px' }}>Transformer Token Visualization</h1>

          <section className="visualizer-controls">
            <div className="visualizer-controls-header">
              <div><strong>Visualization Controls</strong><small>Load a model, run inference, and select the token/layer to inspect.</small></div>
              <span className={`ws-status ${status}`}>WS: {status}</span>
            </div>
            {loading && <div className="control-notice">⏳ Processing…</div>}
            {error && <div className="control-error">❌ {error}</div>}
            <div className="model-control-row">
              <label htmlFor="model-path">Model Path</label>
              <input id="model-path" type="text" value={modelPath} onChange={(e) => setModelPath(e.target.value)} placeholder="Leave blank for a dummy model, or enter an absolute .pth path" />
              <button className="primary" onClick={handleLoadModel}>Load Model</button>
              <button className="success" onClick={handleRegisterHooks}>Register Hooks</button>
              <button className="warning" onClick={handleRunForward}>Run Forward</button>
            </div>
            {modelLoaded && <div className="model-ready">✓ Model loaded and hooks registered</div>}
            <div className="selector-grid">
              <TokenSelector tokenCount={tokenCount} selectedToken={selectedToken} onSelect={setSelectedToken} />
              <LayerSlider maxLayer={maxLayer} currentLayer={currentLayer} onChange={setCurrentLayer} />
            </div>
          </section>

          <BinarySampleLoader
            modelShape={modelInputSchema}
            onFeatureLoaded={handleBinaryFeature}
            onTenhouUrl={(url) => { setBoardUrl(url); if (url) setBoardOpen(true) }}
          />

          <section style={{ marginBottom: '20px', paddingBottom: '20px', borderBottom: '1px solid #dee2e6' }}>
            <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: '12px', marginBottom: '8px', flexWrap: 'wrap' }}>
              <label htmlFor="custom-feature-json" style={{ fontWeight: 600 }}>Custom Feature (JSON)</label>
              <span id="custom-feature-schema" style={{ color: '#586069', fontSize: '13px', fontFamily: 'monospace' }}>
                {modelInputSchema
                  ? `${modelInputSchema.dtype} · [${modelInputSchema.seq_len}, ${modelInputSchema.feature_dim}] or [1, ${modelInputSchema.seq_len}, ${modelInputSchema.feature_dim}]`
                  : 'Input schema unavailable'}
              </span>
            </div>
            <textarea
              id="custom-feature-json"
              ref={customFeatureTextRef}
              disabled={!modelInputSchema}
              aria-describedby="custom-feature-schema custom-feature-error"
              aria-invalid={customFeatureError ? true : undefined}
              onInput={() => { if (customFeatureError) setCustomFeatureError(null) }}
              placeholder={modelInputSchema
                ? `[[0.0, ... ${modelInputSchema.feature_dim} values], ... ${modelInputSchema.seq_len} rows]`
                : 'Load a .pth model to infer the feature shape'}
              spellCheck={false}
              style={{
                width: '100%',
                minHeight: '132px',
                resize: 'vertical',
                padding: '10px',
                boxSizing: 'border-box',
                border: `1px solid ${customFeatureError ? '#dc3545' : '#ced4da'}`,
                borderRadius: '4px',
                fontFamily: 'monospace',
                fontSize: '12px',
                lineHeight: 1.5,
                backgroundColor: modelInputSchema ? '#fff' : '#f1f3f5'
              }}
            />
            {customFeatureError && (
              <div id="custom-feature-error" role="alert" style={{ marginTop: '6px', color: '#c92a2a', fontSize: '13px' }}>
                {customFeatureError}
              </div>
            )}
          </section>

          <div className="visualization-grid">
            <div className="raw-input-panel" style={{ border: '1px solid #dee2e6', borderRadius: '8px' }}>
              <InputFeatures features={inputFeatures} tokenIndex={selectedToken} />
            </div>
            <div className="layer-input-panel" style={{ border: '1px solid #dee2e6', borderRadius: '8px' }}>
              <LayerInputView vector={layerInput} layerIndex={currentLayer} tokenIndex={selectedToken} modelDimension={modelDimension} />
            </div>
            <div className="embedding-panel" style={{ border: '1px solid #dee2e6', borderRadius: '8px', overflow: 'hidden' }}>
              <EmbeddingView embedding={embedding} tokenIndex={selectedToken} />
            </div>
            <div style={{ border: '1px solid #dee2e6', borderRadius: '8px', overflow: 'hidden' }}>
              <AttentionHeatmap attention={attention} tokenIndex={selectedToken} />
            </div>
            <div style={{ border: '1px solid #dee2e6', borderRadius: '8px', overflow: 'hidden' }}>
              <OutputView logits={output} schema={modelOutputSchema} />
            </div>
          </div>
        </div>
      </main>
      <div className={`board-drawer ${currentView === 'visualize' && boardOpen ? 'open' : ''}`} aria-hidden={currentView !== 'visualize' || !boardOpen}>
        <button className="board-backdrop" aria-label="Close board panel" onClick={() => setBoardOpen(false)} />
        {isResizingBoard && <div
          aria-hidden="true"
          onPointerMove={moveBoardResize}
          onPointerUp={endBoardResize}
          onPointerCancel={endBoardResize}
          style={{ position: 'fixed', inset: 0, zIndex: 9999, cursor: 'col-resize', touchAction: 'none' }}
        />}
        <div
          ref={boardSeparatorRef}
          className={`board-separator ${isResizingBoard ? 'resizing' : ''}`}
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize Current Tenhou Board"
          aria-valuemin={300}
          aria-valuemax={1200}
          aria-valuenow={Math.round(boardWidth)}
          onPointerDown={startBoardResize}
          onPointerMove={moveBoardResize}
          onPointerUp={endBoardResize}
          onPointerCancel={endBoardResize}
          onLostPointerCapture={endBoardResize}
          style={{ right: boardWidth - 4 }}
        />
        <aside ref={boardPaneRef} className="board-panel" style={{ width: boardWidth }}>
          <header>
            <div><strong>Current Tenhou Board</strong><small>Interactive sample preview</small></div>
            {boardUrl && <button className="board-external" onClick={() => window.open(boardUrl, '_blank', 'noopener,noreferrer')}>外部開啟 ↗</button>}
            <button className="board-close" aria-label="Close board panel" onClick={() => setBoardOpen(false)}>×</button>
          </header>
          {boardUrl
            ? <iframe key={boardUrl} src={boardUrl} title="Current Tenhou board" style={{ flex: 1, width: '100%', border: 0 }} />
            : <div style={{ padding: 20, color: '#868e96' }}>載入 `.bin` 樣本後會在此顯示實際盤面。</div>}
        </aside>
      </div>
      {currentView === 'visualize' && <VisualizationChatSidebar
        open={chatOpen}
        onClose={() => setChatOpen(false)}
        pageCtx={{
          modelPath,
          modelInputSchema,
          selectedToken,
          currentLayer,
          boardUrl,
          featureMatrix: featureMatrixRef.current,
          validLen: sampleMeta?.valid_len !== undefined ? Number(sampleMeta.valid_len) : null,
          observerSeat: sampleMeta?.player_id !== undefined ? Number(sampleMeta.player_id) : null,
          targetRelativeIndex: sampleMeta?.pred_id !== undefined ? Number(sampleMeta.pred_id) : null,
          targetSeat: sampleMeta?.target_seat !== undefined ? Number(sampleMeta.target_seat) : null,
          seatEncoding: sampleMeta?.seat_encoding === 'absolute_with_relative_target_index' ? sampleMeta.seat_encoding : null,
          oracleLabel: sampleMeta?.label !== undefined ? Number(sampleMeta.label) : null,
        }}
        outputLogits={output}
        outputSchema={modelOutputSchema}
      />}
      {currentView === 'visualize' && !boardOpen && <button
        className="board-toggle"
        onClick={() => setBoardOpen(true)}
      ><span>盤面</span><small>Tenhou</small><b>◀</b></button>}
      {currentView === 'visualize' && !chatOpen && <button
        className="chat-toggle"
        onClick={() => setChatOpen(true)}
      ><span>Chat</span><small>AI</small><b>◀</b></button>}
    </div>
  )
}

export default App
