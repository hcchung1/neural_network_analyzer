import { useState, useRef, useEffect, useCallback } from 'react'
import type { ModelOutputSchema } from './OutputView'

const API_BASE_URL = ''

interface ChatMsg {
  role: 'user' | 'assistant'
  content: string
}

interface PageCtx {
  modelPath: string
  modelInputSchema: { seq_len: number; feature_dim: number } | null
  selectedToken: number
  currentLayer: number
  boardUrl: string
  featureMatrix: number[][] | null
  validLen: number | null
  observerSeat: number | null
  targetRelativeIndex: number | null
  targetSeat: number | null
  seatEncoding: 'absolute_with_relative_target_index' | null
  oracleLabel: number | null
}

interface TenpaiAssessment {
  verdict: string
  tenpai_probability: number
  confidence: string
  evidence: string[]
  counter_evidence: string[]
  missing_information: string[]
  explanation_zh_tw: string
}

interface Props {
  open: boolean
  onClose: () => void
  pageCtx: PageCtx
  outputLogits: number[] | null
  outputSchema: ModelOutputSchema | null
}

export default function VisualizationChatSidebar({ open, onClose, pageCtx, outputLogits, outputSchema }: Props) {
  const [messages, setMessages] = useState<ChatMsg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [tenpaiResult, setTenpaiResult] = useState<TenpaiAssessment | null>(null)
  const [tenpaiText, setTenpaiText] = useState('')
  const bodyRef = useRef<HTMLDivElement>(null)

  const requestPageContext = useCallback(() => ({
    model_path: pageCtx.modelPath,
    model_input_schema: pageCtx.modelInputSchema,
    selected_token: pageCtx.selectedToken,
    current_layer: pageCtx.currentLayer,
    board_url: pageCtx.boardUrl,
    output_logits: outputLogits,
    output_schema: outputSchema,
    feature: pageCtx.featureMatrix,
    valid_len: pageCtx.validLen,
    observer_seat: pageCtx.observerSeat,
    target_relative_index: pageCtx.targetRelativeIndex,
    target_seat: pageCtx.targetSeat,
    seat_encoding: pageCtx.seatEncoding,
    oracle_label: pageCtx.oracleLabel,
  }), [pageCtx, outputLogits, outputSchema])

  const readResponse = useCallback(async (response: Response) => {
    const data = await response.json().catch(() => ({}))
    if (!response.ok || !data.success) throw new Error(data.detail || data.error || `Request failed (${response.status})`)
    return data
  }, [])

  useEffect(() => {
    if (bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight
  }, [messages])

  const submitChat = useCallback(async () => {
    const text = input.trim()
    if (!text || busy) return
    setInput('')
    setBusy(true)
    setError('')
    const newMsgs: ChatMsg[] = [...messages, { role: 'user', content: text }]
    setMessages(newMsgs)
    try {
      const response = await fetch(`${API_BASE_URL}/api/visualize_chat/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: 'chat',
          messages: newMsgs,
          page_context: requestPageContext(),
        }),
      })
      const data = await readResponse(response)
      setMessages((prev) => [...prev, { role: 'assistant', content: data.reply }])
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }, [input, busy, messages, requestPageContext, readResponse])

  const quickTenpai = useCallback(async () => {
    if (busy) return
    setBusy(true)
    setError('')
    setTenpaiResult(null)
    setTenpaiText('')
    try {
      const response = await fetch(`${API_BASE_URL}/api/visualize_chat/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: 'tenpai_quick',
          messages: [],
          page_context: requestPageContext(),
        }),
      })
      const data = await readResponse(response)
      if (data.kind === 'tenpai_assessment') {
        if (data.assessment) {
          setTenpaiResult(data.assessment)
          const explanation = data.assessment.explanation_zh_tw || (data.assessment.verdict ? `判定：${data.assessment.verdict}（機率 ${(data.assessment.tenpai_probability * 100).toFixed(1)}%、信心 ${data.assessment.confidence}）` : '')
          const modelLine = data.model_assessment?.available
            ? `\n\nTransformer：${data.model_assessment.prediction}（${(data.model_assessment.tenpai_probability * 100).toFixed(1)}%，${data.model_assessment.output_type}）`
            : '\n\nTransformer：輸出語意不足，未轉換為聽牌機率。'
          const agreementLine = data.agreement_analysis?.agree === null
            ? '\n盤面與模型：盤面為 uncertain，無法判定一致性。'
            : `\n盤面與模型：${data.agreement_analysis?.agree ? '一致' : '不一致'}。`
          const oracleLine = data.oracle_evaluation
            ? `\nOracle（LLM 完成後由後端比對）：${data.oracle_evaluation.oracle}。`
            : ''
          setMessages((prev) => [
            ...prev,
            { role: 'assistant', content: `🔍 **獨立盤面分析**\n\n${explanation}${modelLine}${agreementLine}${oracleLine}` },
          ])
        } else if (data.text) {
          setTenpaiText(data.text)
          setMessages((prev) => [...prev, { role: 'assistant', content: `🔍 **聽牌分析**\n\n${data.text}` }])
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }, [busy, requestPageContext, readResponse])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submitChat()
    }
  }

  return (
    <div className={`chat-drawer ${open ? 'open' : ''}`} aria-hidden={!open}>
      <button className="chat-backdrop" aria-label="Close chat panel" onClick={onClose} />
      <aside className="chat-panel">
        <header>
          <div><strong>Analysis Chat</strong><small>Scoped to visualization page</small></div>
          <button className="chat-close" aria-label="Close chat panel" onClick={onClose}>×</button>
        </header>
        <div className="chat-toolbar">
          <button
            className="chat-tenpai-btn"
            onClick={quickTenpai}
            disabled={busy || !pageCtx.featureMatrix || pageCtx.validLen === null || pageCtx.targetSeat === null}
            title={pageCtx.validLen !== null && pageCtx.targetSeat !== null ? 'Analyze current binary sample without revealing label to the LLM' : 'Load a binary sample with valid length and seat metadata first'}
          >
            {busy ? '⏳' : '🔍'} 聽牌分析
          </button>
          {tenpaiResult && (
            <div className="chat-tenpai-badge">
              <span>{tenpaiResult.verdict === 'tenpai' ? '✅' : tenpaiResult.verdict === 'not_tenpai' ? '❌' : '❓'}</span>
              <span>{tenpaiResult.verdict}</span>
              <span>{(tenpaiResult.tenpai_probability * 100).toFixed(0)}%</span>
              <span>{tenpaiResult.confidence}</span>
            </div>
          )}
          {tenpaiText && !tenpaiResult && <div className="chat-tenpai-badge"><span>📋</span><span>Analysis</span></div>}
        </div>
        <div ref={bodyRef} className="chat-body">
          {messages.length === 0 && <div className="chat-empty">Binary samples enable blind board/tenpai analysis with valid-length and seat checks. Custom features remain available for discussing model inputs and outputs.</div>}
          {messages.map((msg, i) => (
            <div key={i} className={`chat-msg ${msg.role}`}>
              <div className="chat-msg-content">{msg.content}</div>
            </div>
          ))}
          {busy && <div className="chat-typing">思考中…</div>}
          {error && <div className="chat-error">{error}</div>}
        </div>
        <div className="chat-input-row">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="討論 token、attention、output 與聽牌證據…"
            disabled={busy}
            rows={2}
          />
          <button onClick={submitChat} disabled={busy || !input.trim()}>→</button>
        </div>
      </aside>
    </div>
  )
}
