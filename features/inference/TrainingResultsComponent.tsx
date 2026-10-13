import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import dynamic from 'next/dynamic'
import { Check, Copy } from 'lucide-react'
import { copyImageToClipboard, copyToClipboard } from '@/lib/clipboard'
import { mapWithConcurrency } from '@/lib/performance'
import { readTrainingResults, writeTrainingResults } from '@/lib/training-results-storage'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

interface TrainingResult {
  id: string
  name: string
  color: string
  copyColor: string
  folderPath: string
  visible: boolean
  data: TrainingHistoryData | null
  phaseData: PhaseAnalysisData | null
  randomPhaseData: PhaseAnalysisData | null
  turnMetrics: TurnMetricsData | null
  randomTurnMetrics: TurnMetricsData | null
}

interface TrainingHistoryData {
  epochs: number[]
  trainLoss: (number | null)[]
  valLoss: (number | null)[]
  valAccuracy: (number | null)[]
  valF1: (number | null)[]
  valAuc: (number | null)[]
  valCopyLoss: (number | null)[]
  valCopyAccuracy: (number | null)[]
  valCopyF1: (number | null)[]
  valCopyAuc: (number | null)[]
}

type PhaseName = 'early' | 'mid' | 'late'

interface PhaseMetrics {
  phase: PhaseName
  totalSamples: number
  accuracy: number | null
  precision: number | null
  recall: number | null
  f1Score: number | null
  auc: number | null
  avgProbability: number | null
}

type PhaseAnalysisData = Record<PhaseName, PhaseMetrics>

interface TurnMetricsData {
  turn: number[]
  count: (number | null)[]
  avgProbability: (number | null)[]
  accuracy: (number | null)[]
  precision: (number | null)[]
  recall: (number | null)[]
  f1Score: (number | null)[]
  auc: (number | null)[]
  predictedPositiveRate: (number | null)[]
  positiveRate: (number | null)[]
}

interface TrainingFile {
  name: string
  path: string
  size: number
}

interface FolderScanResponse {
  success: boolean
  folder_path: string
  files: TrainingFile[]
  error?: string
}

interface ReadFileResponse {
  success: boolean
  content: string
  error?: string
}

const COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]
const API_BASE_URL = ''
const STORAGE_KEY = 'training_results'
const PHASES: PhaseName[] = ['early', 'mid', 'late']
const PLOT_CONFIG = { responsive: true, displaylogo: false }
const PLOT_STYLE = { width: '100%', height: '400px' }
const COMBINED_TURN_PLOT_STYLE = { width: '100%', height: '780px' }
const COLOR_UPDATE_DEBOUNCE_MS = 300
const DEFAULT_FOLDER_PATH = '/workspace/Study/Mahjong-Hidden-Information-Forecast/Transformer/output/'

const ResultCopyButton = memo(function ResultCopyButton({ name }: { name: string }) {
  const [copyStatus, setCopyStatus] = useState<'idle' | 'copied' | 'failed'>('idle')
  const resetTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => () => {
    if (resetTimer.current) clearTimeout(resetTimer.current)
  }, [])

  const copyName = async () => {
    if (resetTimer.current) clearTimeout(resetTimer.current)
    try {
      await copyToClipboard(name)
      setCopyStatus('copied')
    } catch {
      setCopyStatus('failed')
    }
    resetTimer.current = setTimeout(() => setCopyStatus('idle'), 2000)
  }

  const copyLabel = copyStatus === 'copied' ? '已複製' : '複製完整名稱'
  return (
    <div style={{ display: 'flex', position: 'relative' }}>
      <button
        type="button"
        aria-label={copyLabel}
        title={copyLabel}
        onClick={() => void copyName()}
        style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: '32px', height: '32px', padding: '5px', border: '1px solid #cbd5e1', borderRadius: '8px', backgroundColor: '#fff', color: '#475569', cursor: 'pointer' }}
      >
        {copyStatus === 'copied' ? <Check size={16} aria-hidden="true" /> : <Copy size={16} aria-hidden="true" />}
      </button>
      <span role="status" className={copyStatus === 'failed' ? 'absolute right-0 top-full whitespace-nowrap text-xs text-red-700' : 'sr-only'}>
        {copyStatus === 'copied' ? '已複製完整名稱' : copyStatus === 'failed' ? '複製失敗，請再試一次' : ''}
      </span>
    </div>
  )
})

const ResultName = memo(function ResultName({ name }: { name: string }) {
  // Keep the final date/time and any trailing suffix intact while the prefix shrinks.
  const dates = Array.from(name.matchAll(/\d{8}(?:[_-]\d{6})?|\d{4}[-_]\d{2}[-_]\d{2}/g))
  const suffixStart = dates.length ? dates[dates.length - 1].index! : Math.max(0, name.length - 16)
  return (
    <strong title={name} style={{ display: 'flex', minWidth: 0 }}>
      <span style={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{name.slice(0, suffixStart)}</span>
      <span data-name-suffix style={{ flexShrink: 0, whiteSpace: 'nowrap' }}>{name.slice(suffixStart)}</span>
    </strong>
  )
})

function ResultImageCopyButton({ name, folderPath, kind }: { name: string; folderPath: string; kind: 'training' | 'phase' }) {
  const [availableImage, setAvailableImage] = useState<string | null>(null)
  const [status, setStatus] = useState<'idle' | 'loading' | 'copied' | 'failed'>('idle')
  const [error, setError] = useState('')
  const resetTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => () => {
    if (resetTimer.current) clearTimeout(resetTimer.current)
  }, [])
  const imageUrl = `/api/v1/training-images?${new URLSearchParams({ folder: folderPath, run: name, kind })}`
  useEffect(() => {
    const controller = new AbortController()
    setAvailableImage(null)
    void fetch(`${imageUrl}&check=1`, { cache: 'no-store', signal: controller.signal })
      .then(response => response.ok ? response.json() : null)
      .then(data => {
        if (!controller.signal.aborted && data?.exists === true) setAvailableImage(imageUrl)
      })
      .catch(() => {})
    return () => controller.abort()
  }, [imageUrl])
  const plotName = kind === 'training' ? '訓練圖片' : 'Phase圖片'
  const label = `複製${plotName}`
  const copyImage = async () => {
    if (resetTimer.current) clearTimeout(resetTimer.current)
    setStatus('loading')
    setError('')
    try {
      await copyImageToClipboard(imageUrl)
      setStatus('copied')
      resetTimer.current = setTimeout(() => setStatus('idle'), 2000)
    } catch (error) {
      setStatus('failed')
      setError(error instanceof Error ? error.message : '圖片複製失敗')
    }
  }
  if (availableImage !== imageUrl) return null
  return (
    <div style={{ minWidth: 0 }}>
      <button type="button" aria-label={label} title={label} disabled={status === 'loading'}
        onClick={() => void copyImage()}
        className="training-image-copy-button"
      >
        {status === 'copied' ? <Check size={14} aria-hidden="true" /> : <Copy size={14} aria-hidden="true" />}
        <span>{status === 'loading' ? `複製中 (${plotName})` : status === 'copied' ? `已複製 (${plotName})` : label}</span>
      </button>
      <span role="status" className={status === 'failed' ? 'block mt-1 text-xs text-red-700' : 'sr-only'}>
        {status === 'copied' ? `${label}：已複製` : error}
      </span>
    </div>
  )
}

interface FullscreenPlotProps {
  data: any[]
  layout: any
  style?: React.CSSProperties
}

function FullscreenPlot({ data, layout, style = PLOT_STYLE }: FullscreenPlotProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [isFullscreen, setIsFullscreen] = useState(false)

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(document.fullscreenElement === containerRef.current)
      requestAnimationFrame(() => window.dispatchEvent(new Event('resize')))
    }
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  const toggleFullscreen = async () => {
    if (document.fullscreenElement === containerRef.current) {
      await document.exitFullscreen()
    } else {
      await containerRef.current?.requestFullscreen()
    }
  }

  const fullscreenIcon = isFullscreen 
    ? {
        width: 24, height: 24,
        path: 'M5 16h3v3h2v-5H5v2zm3-8H5v2h5V5H8v3zm6 11h2v-3h3v-2h-5v5zm2-11V5h-2v5h5V8h-3z'
      }
    : {
        width: 24, height: 24,
        path: 'M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z'
      };

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        border: '1px solid #e5e5e5',
        borderRadius: isFullscreen ? 0 : '8px',
        padding: isFullscreen ? '4px' : '8px',
        backgroundColor: '#fff',
        boxSizing: 'border-box',
        width: '100%',
        height: isFullscreen ? '100vh' : (style?.height || '400px'),
        overflow: 'hidden'
      }}
    >
      <style>{`
        .js-plotly-plot .plotly .modebar {
          display: flex !important;
          flex-direction: row !important;
          right: 5px !important;
          top: -5px !important;
        }
        .js-plotly-plot .plotly .modebar-group {
          display: flex !important;
          flex-direction: row !important;
          margin-bottom: 0 !important;
        }
      `}</style>
      <Plot
        data={data}
        layout={{ 
          ...layout,
          modebar: { orientation: 'h' as const }
        } as any}
        config={{ 
          ...PLOT_CONFIG, 
          displayModeBar: true, 
          displaylogo: false,
          modeBarButtonsToAdd: [
            {
              name: isFullscreen ? 'Exit Fullscreen' : 'Fullscreen',
              title: isFullscreen ? '縮小 (Exit Fullscreen)' : '全螢幕 (Fullscreen)',
              icon: fullscreenIcon,
              click: toggleFullscreen
            }
          ]
        }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
      />
    </div>
  )
}

type ResultColorField = 'color' | 'copyColor'

interface ResultColorInputProps {
  resultId: string
  field: ResultColorField
  value: string
  title: string
  onCommit: (id: string, field: ResultColorField, value: string) => void
}

const ResultColorInput = memo(function ResultColorInput({
  resultId,
  field,
  value,
  title,
  onCommit
}: ResultColorInputProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const latestValueRef = useRef(value)
  const commitTimerRef = useRef<number | null>(null)

  const clearCommitTimer = useCallback(() => {
    if (commitTimerRef.current !== null) {
      window.clearTimeout(commitTimerRef.current)
      commitTimerRef.current = null
    }
  }, [])

  const commitLatestValue = useCallback(() => {
    clearCommitTimer()
    onCommit(resultId, field, latestValueRef.current)
  }, [clearCommitTimer, field, onCommit, resultId])

  const handleInput = useCallback((event: React.FormEvent<HTMLInputElement>) => {
    latestValueRef.current = event.currentTarget.value
    clearCommitTimer()
    commitTimerRef.current = window.setTimeout(commitLatestValue, COLOR_UPDATE_DEBOUNCE_MS)
  }, [clearCommitTimer, commitLatestValue])

  useEffect(() => {
    latestValueRef.current = value
    if (inputRef.current) inputRef.current.value = value
  }, [value])

  useEffect(() => clearCommitTimer, [clearCommitTimer])

  return (
    <input
      ref={inputRef}
      type="color"
      defaultValue={value}
      onInput={handleInput}
      onBlur={commitLatestValue}
      title={title}
      aria-label={title}
      style={{ width: '24px', height: '24px', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
    />
  )
})

function parseNumber(value: string | undefined): number | null {
  if (value === undefined || value.trim() === '') return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function splitCsvLine(line: string): string[] {
  const result: string[] = []
  let current = ''
  let inQuotes = false

  for (let i = 0; i < line.length; i++) {
    const char = line[i]
    const next = line[i + 1]
    if (char === '"' && inQuotes && next === '"') {
      current += '"'
      i++
    } else if (char === '"') {
      inQuotes = !inQuotes
    } else if (char === ',' && !inQuotes) {
      result.push(current)
      current = ''
    } else {
      current += char
    }
  }
  result.push(current)
  return result.map(col => col.trim())
}

function parseTrainingHistoryCSV(csvText: string): TrainingHistoryData {
  const lines = csvText.trim().split('\n')
  const data: TrainingHistoryData = {
    epochs: [],
    trainLoss: [],
    valLoss: [],
    valAccuracy: [],
    valF1: [],
    valAuc: [],
    valCopyLoss: [],
    valCopyAccuracy: [],
    valCopyF1: [],
    valCopyAuc: []
  }

  const historyMarkerIdx = lines.findIndex(line => line.trim() === '[TRAINING_HISTORY]')
  if (historyMarkerIdx === -1 || historyMarkerIdx + 1 >= lines.length) return data

  const header = splitCsvLine(lines[historyMarkerIdx + 1].trim())
  const idx = (name: string) => header.indexOf(name)
  const epochIdx = idx('epoch')
  if (epochIdx === -1) return data

  const trainLossIdx = idx('train_loss')
  const valLossIdx = idx('val_loss')
  const valAccuracyIdx = idx('val_accuracy')
  const valF1Idx = idx('val_f1')
  const valAucIdx = idx('val_auc')
  const valCopyLossIdx = idx('val_copy_loss')
  const valCopyAccuracyIdx = idx('val_copy_accuracy')
  const valCopyF1Idx = idx('val_copy_f1')
  const valCopyAucIdx = idx('val_copy_auc')

  for (let i = historyMarkerIdx + 2; i < lines.length; i++) {
    const line = lines[i].trim()
    if (!line || line.startsWith('[')) break

    const cols = splitCsvLine(line)
    const epoch = parseNumber(cols[epochIdx])
    if (epoch === null) continue

    data.epochs.push(epoch)
    data.trainLoss.push(parseNumber(cols[trainLossIdx]))
    data.valLoss.push(parseNumber(cols[valLossIdx]))
    data.valAccuracy.push(parseNumber(cols[valAccuracyIdx]))
    data.valF1.push(parseNumber(cols[valF1Idx]))
    data.valAuc.push(parseNumber(cols[valAucIdx]))
    data.valCopyLoss.push(parseNumber(cols[valCopyLossIdx]))
    data.valCopyAccuracy.push(parseNumber(cols[valCopyAccuracyIdx]))
    data.valCopyF1.push(parseNumber(cols[valCopyF1Idx]))
    data.valCopyAuc.push(parseNumber(cols[valCopyAucIdx]))
  }

  return data
}

function parsePhaseAnalysisCSV(csvText: string): PhaseAnalysisData | null {
  const lines = csvText.trim().split('\n').map(line => line.trim()).filter(Boolean)
  if (lines.length < 2) return null

  const header = splitCsvLine(lines[0])
  const phaseIdx = header.indexOf('phase')
  if (phaseIdx === -1) return null

  const idx = (name: string) => header.indexOf(name)
  const totalSamplesIdx = idx('total_samples')
  const accuracyIdx = idx('accuracy')
  const precisionIdx = idx('precision')
  const recallIdx = idx('recall')
  const f1Idx = idx('f1_score')
  const aucIdx = idx('auc')
  const avgProbabilityIdx = idx('avg_probability')
  const data: Partial<PhaseAnalysisData> = {}

  for (const line of lines.slice(1)) {
    const cols = splitCsvLine(line)
    const phase = cols[phaseIdx] as PhaseName
    if (!PHASES.includes(phase)) continue

    data[phase] = {
      phase,
      totalSamples: parseNumber(cols[totalSamplesIdx]) ?? 0,
      accuracy: parseNumber(cols[accuracyIdx]),
      precision: parseNumber(cols[precisionIdx]),
      recall: parseNumber(cols[recallIdx]),
      f1Score: parseNumber(cols[f1Idx]),
      auc: parseNumber(cols[aucIdx]),
      avgProbability: parseNumber(cols[avgProbabilityIdx])
    }
  }

  if (!PHASES.some(phase => data[phase])) return null

  return PHASES.reduce((acc, phase) => {
    acc[phase] = data[phase] ?? {
      phase,
      totalSamples: 0,
      accuracy: null,
      precision: null,
      recall: null,
      f1Score: null,
      auc: null,
      avgProbability: null
    }
    return acc
  }, {} as PhaseAnalysisData)
}

function parseTurnMetricsCSV(csvText: string): TurnMetricsData | null {
  const lines = csvText.trim().split('\n').map(line => line.trim()).filter(Boolean)
  if (lines.length < 2) return null

  const header = splitCsvLine(lines[0])
  const idx = (name: string) => header.indexOf(name)
  const turnIdx = idx('turn')
  if (turnIdx === -1) return null

  const countIdx = idx('count')
  const avgProbabilityIdx = idx('avg_probability')
  const accuracyIdx = idx('accuracy')
  const precisionIdx = idx('precision')
  const recallIdx = idx('recall')
  const f1Idx = idx('f1_score')
  const aucIdx = idx('auc')
  const predictedPositiveRateIdx = idx('predicted_positive_rate')
  const positiveRateIdx = idx('positive_rate')

  const data: TurnMetricsData = {
    turn: [],
    count: [],
    avgProbability: [],
    accuracy: [],
    precision: [],
    recall: [],
    f1Score: [],
    auc: [],
    predictedPositiveRate: [],
    positiveRate: []
  }

  for (const line of lines.slice(1)) {
    const cols = splitCsvLine(line)
    const turn = parseNumber(cols[turnIdx])
    if (turn === null) continue
    data.turn.push(turn)
    data.count.push(parseNumber(cols[countIdx]))
    data.avgProbability.push(parseNumber(cols[avgProbabilityIdx]))
    data.accuracy.push(parseNumber(cols[accuracyIdx]))
    data.precision.push(parseNumber(cols[precisionIdx]))
    data.recall.push(parseNumber(cols[recallIdx]))
    data.f1Score.push(parseNumber(cols[f1Idx]))
    data.auc.push(parseNumber(cols[aucIdx]))
    data.predictedPositiveRate.push(parseNumber(cols[predictedPositiveRateIdx]))
    data.positiveRate.push(parseNumber(cols[positiveRateIdx]))
  }

  return data.turn.length > 0 ? data : null
}

function normalizeResultName(fileName: string, filePath?: string): string {
  if (fileName === 'comparison_table.csv' && filePath) {
    const folderName = filePath.split(/[\\/]/).slice(-2, -1)[0]
    return folderName || fileName.replace(/\.csv$/, '')
  }

  return fileName
    .replace(/_history\.csv$/, '')
    .replace(/_base_results_phase_summary\.csv$/, '')
    .replace(/_random_feature_results_phase_summary\.csv$/, '')
    .replace(/_base_results_turn_metrics\.csv$/, '')
    .replace(/_random_feature_results_turn_metrics\.csv$/, '')
    .replace(/_phase_summary\.csv$/, '')
    .replace(/\.csv$/, '')
}

function getChartDisplayName(name: string): string {
  const match = name.match(/\d{8}_\d{6}/)
  return match ? match[0] : name
}

function restoreHistoryData(data: TrainingHistoryData | null): TrainingHistoryData | null {
  if (!data) return null
  const emptyAuc = data.epochs.map(() => null)
  return {
    ...data,
    valAuc: data.valAuc ?? emptyAuc,
    valCopyAuc: data.valCopyAuc ?? emptyAuc
  }
}

function restorePhaseData(data: PhaseAnalysisData | null): PhaseAnalysisData | null {
  if (!data) return null
  return PHASES.reduce((restored, phase) => {
    restored[phase] = { ...data[phase], auc: data[phase]?.auc ?? null }
    return restored
  }, {} as PhaseAnalysisData)
}

function restoreTurnMetrics(data: TurnMetricsData | null): TurnMetricsData | null {
  if (!data) return null
  return { ...data, auc: data.auc ?? data.turn.map(() => null) }
}

async function restoreResults(): Promise<TrainingResult[]> {
  try {
    if (typeof window === 'undefined') return []
    const parsed = await readTrainingResults<TrainingResult>(STORAGE_KEY)
    if (parsed.length) {
      if (Array.isArray(parsed)) {
        return parsed.map(result => ({
          ...result,
          data: restoreHistoryData(result.data ?? null),
          phaseData: restorePhaseData(result.phaseData ?? null),
          randomPhaseData: restorePhaseData(result.randomPhaseData ?? null),
          turnMetrics: restoreTurnMetrics(result.turnMetrics ?? null),
          randomTurnMetrics: restoreTurnMetrics(result.randomTurnMetrics ?? null)
        }))
      }
    }
  } catch (e) {
    console.error('Failed to restore training results:', e)
    throw e
  }
  return []
}

interface TrainingResultsPageProps {
  onBack: () => void
  requestedFolder?: { path: string; requestId: number } | null
}

export default function TrainingResultsPage({ onBack, requestedFolder }: TrainingResultsPageProps) {
  const [results, setResults] = useState<TrainingResult[]>([])
  const [isLoaded, setIsLoaded] = useState(false)

  const pendingPersistence = useRef<TrainingResult[] | null>(null)
  const persistenceReady = useRef(false)
  useEffect(() => {
    let active = true
    void restoreResults().then(restored => {
      if (active) { persistenceReady.current = true; setResults(restored) }
    })
      .catch(error => { if (active) setScanError(`Unable to restore training results: ${error.message}`) })
      .finally(() => { if (active) setIsLoaded(true) })
    return () => { active = false }
  }, [])
  const [folderInput, setFolderInput] = useState(DEFAULT_FOLDER_PATH)
  const [loading, setLoading] = useState(false)
  const [scanError, setScanError] = useState<string | null>(null)

  useEffect(() => {
    if (!isLoaded || !persistenceReady.current) return
    pendingPersistence.current = results
    const timer = setTimeout(() => {
      pendingPersistence.current = null
      void writeTrainingResults(STORAGE_KEY, results).catch(error =>
        setScanError(`Unable to save training results: ${error.message}`))
    }, 500)
    return () => clearTimeout(timer)
  }, [results, isLoaded])
  useEffect(() => () => {
    if (pendingPersistence.current) {
      void writeTrainingResults(STORAGE_KEY, pendingPersistence.current).catch(console.error)
    }
  }, [])

  const scanFolder = async (folderOverride?: string) => {
    const targetFolder = (folderOverride ?? folderInput).trim()
    if (!targetFolder) return

    setLoading(true)
    setScanError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/api/legacy/training/scan`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ folder_path: targetFolder }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data: FolderScanResponse = await response.json()

      if (!data.success) {
        setScanError(data.error || 'Failed to scan folder')
        setLoading(false)
        return
      }

      if (data.files.length === 0) {
        setScanError('No training history CSV files (*_history.csv) or phase analysis files (*_phase_summary.csv / comparison_table.csv) found in folder')
        setLoading(false)
        return
      }

      const resultMap = new Map<string, TrainingResult>()
      data.files.forEach((file, i) => {
        const name = normalizeResultName(file.name, file.path)
        resultMap.set(name, {
          id: `${Date.now()}-${i}`,
          name,
          color: COLORS[(results.length + i) % COLORS.length],
          copyColor: '#2ca02c',
          folderPath: targetFolder,
          visible: true,
          data: null,
          phaseData: null,
          randomPhaseData: null,
          turnMetrics: null,
          randomTurnMetrics: null
        })
      })

      const loadedFiles = await mapWithConcurrency(data.files, 4, async file => {
        try {
          const readResponse = await fetch(`${API_BASE_URL}/api/legacy/training/read_file`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ file_path: file.path }),
          })

          if (!readResponse.ok) {
            console.warn(`Failed to read ${file.name}: HTTP ${readResponse.status}`)
            return null
          }

          const fileData: ReadFileResponse = await readResponse.json()
          if (!fileData.success) {
            console.warn(`Failed to read ${file.name}: ${fileData.error}`)
            return null
          }

          return { file, fileData }
        } catch (error) {
          console.error(`Error reading file ${file.name}:`, error)
          return null
        }
      })

      for (const loaded of loadedFiles) {
        if (!loaded) continue
        const { file, fileData } = loaded
        try {
          const name = normalizeResultName(file.name, file.path)
          const result = resultMap.get(name)
          if (!result) continue

          if (file.name.endsWith('_history.csv')) {
            const parsed = parseTrainingHistoryCSV(fileData.content)
            if (parsed.epochs.length > 0) {
              result.data = parsed
            }
          } else if (file.name.endsWith('_phase_summary.csv') || file.name === 'comparison_table.csv') {
            const parsed = parsePhaseAnalysisCSV(fileData.content)
            if (parsed) {
              if (file.name.includes('_random_feature_')) {
                result.randomPhaseData = parsed
              } else {
                result.phaseData = parsed
              }
            }
          } else if (file.name.endsWith('_turn_metrics.csv')) {
            const parsed = parseTurnMetricsCSV(fileData.content)
            if (parsed) {
              if (file.name.includes('_random_feature_')) {
                result.randomTurnMetrics = parsed
              } else {
                result.turnMetrics = parsed
              }
            }
          }
        } catch (err) {
          console.error(`Error processing file ${file.name}:`, err)
        }
      }

      const newResults = Array.from(resultMap.values()).filter(result => result.data || result.phaseData || result.randomPhaseData || result.turnMetrics || result.randomTurnMetrics)

      if (newResults.length > 0) {
        persistenceReady.current = true
        setResults(prev => {
          const updated = [...prev, ...newResults]
          return updated
        })
      } else {
        setScanError('No valid training history or phase analysis CSV files could be parsed')
      }

      setFolderInput('')
    } catch (error) {
      console.error('Error scanning folder:', error)
      setScanError(error instanceof Error ? error.message : 'Failed to scan folder')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!isLoaded || !persistenceReady.current || !requestedFolder?.path) return
    setFolderInput(requestedFolder.path)
    void scanFolder(requestedFolder.path)
    // requestId intentionally permits importing the same folder again.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLoaded, requestedFolder?.requestId])

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      scanFolder()
    }
  }

  const removeResult = (id: string) => {
    setResults(prev => prev.filter(r => r.id !== id))
  }

  const toggleVisibility = (id: string) => {
    setResults(prev => prev.map(r =>
      r.id === id ? { ...r, visible: !r.visible } : r
    ))
  }

  const updateResultColor = useCallback((id: string, field: ResultColorField, value: string) => {
    setResults(prev => {
      const result = prev.find(item => item.id === id)
      if (!result || result[field] === value) return prev
      return prev.map(item => item.id === id ? { ...item, [field]: value } : item)
    })
  }, [])

  const clearAllResults = () => {
    if (confirm('確定要清除所有訓練成果嗎？')) {
      setResults([])
    }
  }

  const visibleResults = useMemo(() => results.filter(r => r.visible && r.data), [results])
  const visiblePhaseResults = useMemo(() => results.filter(r => r.visible && (r.phaseData || r.randomPhaseData)), [results])
  const visibleTurnMetricResults = useMemo(() => results.filter(r => r.visible && (r.turnMetrics || r.randomTurnMetrics)), [results])

  const createTraces = (metricKey: keyof TrainingHistoryData) => {
    return visibleResults.map(result => ({
      x: result.data!.epochs,
      y: result.data![metricKey] as (number | null)[],
      type: 'scatter' as const,
      mode: 'lines+markers' as const,
      name: getChartDisplayName(result.name),
      line: { color: result.color, width: 2 },
      marker: { size: 4 },
      hovertemplate: '%{y:.4f}<extra></extra>'
    }))
  }

  const createCombinedTraces = (normalKey: keyof TrainingHistoryData, copyKey: keyof TrainingHistoryData) => {
    const traces: any[] = []
    visibleResults.forEach(result => {
      traces.push({
        x: result.data!.epochs,
        y: result.data![normalKey] as (number | null)[],
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        name: `${getChartDisplayName(result.name)} (Normal)`,
        line: { color: result.color, width: 2 },
        marker: { size: 4 },
        hovertemplate: '%{y:.4f}<extra></extra>'
      })
      traces.push({
        x: result.data!.epochs,
        y: result.data![copyKey] as (number | null)[],
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        name: `${getChartDisplayName(result.name)} (Copy)`,
        line: { color: result.copyColor, width: 2, dash: 'dash' as const },
        marker: { size: 4 },
        hovertemplate: '%{y:.4f}<extra></extra>'
      })
    })
    return traces
  }

  const createPhaseTraces = (metricKey: keyof Omit<PhaseMetrics, 'phase'>) => {
    const traces: any[] = []
    visiblePhaseResults.forEach(result => {
      if (result.phaseData) {
        traces.push({
          x: PHASES.map(phase => phase.toUpperCase()),
          y: PHASES.map(phase => result.phaseData![phase][metricKey]),
          type: 'bar' as const,
          name: `${getChartDisplayName(result.name)} (Base)`,
          marker: { color: result.color },
          hovertemplate: '%{x}: %{y:.4f}<extra></extra>'
        })
      }
      if (result.randomPhaseData) {
        traces.push({
          x: PHASES.map(phase => phase.toUpperCase()),
          y: PHASES.map(phase => result.randomPhaseData![phase][metricKey]),
          type: 'bar' as const,
          name: `${getChartDisplayName(result.name)} (Random)`,
          marker: { color: result.copyColor },
          hovertemplate: '%{x}: %{y:.4f}<extra></extra>'
        })
      }
    })
    return traces
  }

  const createTurnMetricTraces = (metricKey: keyof Omit<TurnMetricsData, 'turn'>) => {
    const traces: any[] = []
    visibleTurnMetricResults.forEach(result => {
      const prefix = visibleTurnMetricResults.length > 1 ? `${getChartDisplayName(result.name)} ` : ''
      if (result.turnMetrics) {
        traces.push({
          x: result.turnMetrics.turn,
          y: result.turnMetrics[metricKey] as (number | null)[],
          type: 'scatter' as const,
          mode: 'lines+markers' as const,
          name: `${prefix}Base`,
          line: { color: result.color, width: 2 },
          marker: { symbol: 'circle', size: 5 },
          hovertemplate: 'Turn %{x}: %{y:.4f}<extra></extra>'
        })
      }
      if (result.randomTurnMetrics) {
        traces.push({
          x: result.randomTurnMetrics.turn,
          y: result.randomTurnMetrics[metricKey] as (number | null)[],
          type: 'scatter' as const,
          mode: 'lines+markers' as const,
          name: `${prefix}RandomFeature`,
          line: { color: result.copyColor, width: 2, dash: 'dash' as const },
          marker: { symbol: 'x', size: 6 },
          hovertemplate: 'Turn %{x}: %{y:.4f}<extra></extra>'
        })
      }
    })
    return traces
  }

  const createBaseRandomTurnMetricTraces = () => {
    const traces: any[] = []
    visibleTurnMetricResults.forEach(result => {
      const prefix = visibleTurnMetricResults.length > 1 ? `${getChartDisplayName(result.name)} ` : ''

      if (result.turnMetrics?.count.some(v => v !== null)) {
        traces.push({
          x: result.turnMetrics.turn,
          y: result.turnMetrics.count,
          type: 'bar' as const,
          name: `${prefix}Base count`,
          marker: { color: '#999999', opacity: 0.16 },
          width: 0.8,
          yaxis: 'y2',
          hovertemplate: 'Turn %{x}: %{y:,} samples<extra></extra>'
        })
      }

      if (result.turnMetrics) {
        traces.push(
          {
            x: result.turnMetrics.turn,
            y: result.turnMetrics.avgProbability,
            type: 'scatter' as const,
            mode: 'lines+markers' as const,
            name: `${prefix}Base - Confidence`,
            line: { color: '#1f77b4', width: 2 },
            marker: { symbol: 'circle', size: 5 },
            hovertemplate: 'Turn %{x}: %{y:.4f}<extra></extra>'
          },
          {
            x: result.turnMetrics.turn,
            y: result.turnMetrics.predictedPositiveRate,
            type: 'scatter' as const,
            mode: 'lines+markers' as const,
            name: `${prefix}Base - Predicted Positive Rate`,
            line: { color: '#ff7f0e', width: 2 },
            marker: { symbol: 'circle', size: 5 },
            hovertemplate: 'Turn %{x}: %{y:.4f}<extra></extra>'
          },
          {
            x: result.turnMetrics.turn,
            y: result.turnMetrics.positiveRate,
            type: 'scatter' as const,
            mode: 'lines+markers' as const,
            name: `${prefix}Base - Actual Positive Rate`,
            line: { color: '#2ca02c', width: 2 },
            marker: { symbol: 'circle', size: 5 },
            hovertemplate: 'Turn %{x}: %{y:.4f}<extra></extra>'
          }
        )
      }

      if (result.randomTurnMetrics) {
        traces.push(
          {
            x: result.randomTurnMetrics.turn,
            y: result.randomTurnMetrics.avgProbability,
            type: 'scatter' as const,
            mode: 'lines+markers' as const,
            name: `${prefix}RandomFeature - Confidence`,
            line: { color: '#1f77b4', width: 2, dash: 'dash' as const },
            marker: { symbol: 'x', size: 6 },
            hovertemplate: 'Turn %{x}: %{y:.4f}<extra></extra>'
          },
          {
            x: result.randomTurnMetrics.turn,
            y: result.randomTurnMetrics.predictedPositiveRate,
            type: 'scatter' as const,
            mode: 'lines+markers' as const,
            name: `${prefix}RandomFeature - Predicted Positive Rate`,
            line: { color: '#ff7f0e', width: 2, dash: 'dash' as const },
            marker: { symbol: 'x', size: 6 },
            hovertemplate: 'Turn %{x}: %{y:.4f}<extra></extra>'
          },
          {
            x: result.randomTurnMetrics.turn,
            y: result.randomTurnMetrics.positiveRate,
            type: 'scatter' as const,
            mode: 'lines+markers' as const,
            name: `${prefix}RandomFeature - Actual Positive Rate`,
            line: { color: '#2ca02c', width: 2, dash: 'dash' as const },
            marker: { symbol: 'x', size: 6 },
            hovertemplate: 'Turn %{x}: %{y:.4f}<extra></extra>'
          }
        )
      }
    })
    return traces
  }

      const createLayout = (title: string, yaxisTitle: string) => ({
    autosize: true,
    title: { text: title, font: { size: 16 }, x: 0, xanchor: 'left' as const, y: 0.98, yanchor: 'top' as const },
    xaxis: { title: 'Epoch', gridcolor: '#e5e5e5' },
    yaxis: { title: yaxisTitle, gridcolor: '#e5e5e5' },
    plot_bgcolor: '#fafafa',
    paper_bgcolor: '#ffffff',
    hovermode: 'closest' as const,
    showlegend: true,
    legend: { x: 0, y: 1.02, orientation: 'h' as const, xanchor: 'left' as const, yanchor: 'bottom' as const },
    margin: { t: 90, l: 60, r: 20, b: 50 }
  })

      const createPhaseLayout = (title: string, yaxisTitle: string, yMax?: number) => ({
    autosize: true,
    title: { text: title, font: { size: 16 }, x: 0, xanchor: 'left' as const, y: 0.98, yanchor: 'top' as const },
    xaxis: { title: 'Phase', gridcolor: '#e5e5e5' },
    yaxis: { title: yaxisTitle, gridcolor: '#e5e5e5', range: yMax ? [0, yMax] : undefined },
    plot_bgcolor: '#fafafa',
    paper_bgcolor: '#ffffff',
    hovermode: 'closest' as const,
    showlegend: true,
    barmode: 'group' as const,
    legend: { x: 0, y: 1.02, orientation: 'h' as const, xanchor: 'left' as const, yanchor: 'bottom' as const },
    margin: { t: 90, l: 60, r: 20, b: 50 }
  })

    const createBaseRandomTurnMetricsLayout = () => ({
    autosize: true,
    title: { text: 'Turn Metrics Comparison - Base vs RandomFeature', font: { size: 16 }, x: 0, xanchor: 'left' as const, y: 0.99, yanchor: 'top' as const },
    xaxis: { title: 'Turn Number (Total Discards)', gridcolor: '#e5e5e5', range: [0, 78] },
    yaxis: { title: 'Probability / Rate', gridcolor: '#e5e5e5', range: [-0.03, 1.03] },
    yaxis2: { title: 'Sample Count', overlaying: 'y' as const, side: 'right' as const, showgrid: false },
    plot_bgcolor: '#fafafa',
    paper_bgcolor: '#ffffff',
    hovermode: 'closest' as const,
    showlegend: true,
    legend: { x: 0, y: 1.08, xanchor: 'left' as const, yanchor: 'bottom' as const, orientation: 'h' as const },
    margin: { l: 60, r: 80, t: 150, b: 80 },
    shapes: [
      { type: 'rect' as const, xref: 'x' as const, yref: 'paper' as const, x0: 0, x1: 24, y0: 0, y1: 1, fillcolor: 'green', opacity: 0.12, line: { width: 0 }, layer: 'below' as const },
      { type: 'rect' as const, xref: 'x' as const, yref: 'paper' as const, x0: 24, x1: 40, y0: 0, y1: 1, fillcolor: 'gold', opacity: 0.12, line: { width: 0 }, layer: 'below' as const },
      { type: 'rect' as const, xref: 'x' as const, yref: 'paper' as const, x0: 40, x1: 78, y0: 0, y1: 1, fillcolor: 'red', opacity: 0.12, line: { width: 0 }, layer: 'below' as const },
    ]
  })

  if (!isLoaded) return null;
  return (
    <div className="training-results" style={{ padding: '24px', fontFamily: 'system-ui, sans-serif' }}>
      <h1 style={{ marginTop: 0, marginBottom: '24px' }}>
        <button
          onClick={onBack}
          style={{
            marginRight: '12px',
            padding: '6px 12px',
            backgroundColor: '#fff',
            border: '1px solid #cbd5e1',
            borderRadius: '8px',
            cursor: 'pointer'
          }}
        >
          Back
        </button>
        訓練成果比較 (Training Results)
      </h1>

      <div style={{ display: 'flex', gap: '12px', marginBottom: '24px' }}>
        <input
          type="text"
          value={folderInput}
          onChange={(e) => setFolderInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="輸入資料夾路徑（支援 *_history.csv、*_phase_summary.csv、comparison_table.csv）"
          style={{
            flex: 1,
            minWidth: 0,
            padding: '10px',
            fontSize: '14px',
            border: '1px solid #cbd5e1',
            borderRadius: '8px',
            fontFamily: 'monospace'
          }}
        />
        <button
          onClick={() => void scanFolder()}
          disabled={loading}
          style={{
            padding: '10px 20px',
            backgroundColor: '#2563eb',
            color: '#fff',
            border: 'none',
            borderRadius: '8px',
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.7 : 1
          }}
        >
          {loading ? '掃描中...' : '掃描資料夾'}
        </button>
      </div>

      {scanError && (
        <div style={{
          padding: '12px',
          backgroundColor: '#f8d7da',
          color: '#721c24',
          borderRadius: '4px',
          marginBottom: '16px',
          fontSize: '14px'
        }}>
          ❌ {scanError}
        </div>
      )}

      {results.length === 0 ? (
        <div className="training-empty">
          <p className="font-semibold text-slate-700">尚無訓練成果</p>
          <p className="mt-2 text-sm text-slate-500">輸入資料夾路徑後點擊「掃描資料夾」，系統會自動找到訓練曲線與 Phase Analysis CSV 並繪製互動式圖表</p>
        </div>
      ) : (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h3 style={{ margin: 0 }}>已加入的成果 ({results.length} 個)</h3>
            <button
              onClick={clearAllResults}
              style={{
                padding: '6px 12px',
                backgroundColor: '#dc3545',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '14px'
              }}
            >
              清除全部
            </button>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '24px' }}>
            {results.map(result => (
              <div
                key={result.id}
                data-result-card
                className="training-result-card"
                style={{
                  padding: '12px',
                  border: '1px solid #e5e5e5',
                  borderRadius: '8px',
                  backgroundColor: '#ffffff'
                }}
              >
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginRight: '12px' }}>
                  <ResultColorInput
                    resultId={result.id}
                    field="color"
                    value={result.color}
                    title="Normal / Phase color"
                    onCommit={updateResultColor}
                  />
                  <ResultColorInput
                    resultId={result.id}
                    field="copyColor"
                    value={result.copyColor}
                    title="Copy color"
                    onCommit={updateResultColor}
                  />
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <ResultName name={result.name} />
                  <div style={{ fontSize: '12px', color: '#888', wordBreak: 'break-all' }}>{result.folderPath}</div>
                  <div style={{ display: 'flex', gap: '6px', marginTop: '4px', fontSize: '11px' }}>
                    {result.data && <span style={{ padding: '2px 6px', borderRadius: '10px', backgroundColor: '#e7f5ff', color: '#1971c2' }}>Training</span>}
                    {(result.phaseData || result.randomPhaseData) && <span style={{ padding: '2px 6px', borderRadius: '10px', backgroundColor: '#ebfbee', color: '#2b8a3e' }}>Phase Analysis</span>}
                    {(result.turnMetrics || result.randomTurnMetrics) && <span style={{ padding: '2px 6px', borderRadius: '10px', backgroundColor: '#fff4e6', color: '#e8590c' }}>Turn Metrics</span>}
                  </div>
                </div>
                <div className="training-result-actions">
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '8px' }}>
                    <ResultCopyButton name={result.name} />
                    <button
                      onClick={() => toggleVisibility(result.id)}
                      style={{
                        height: '32px',
                        minWidth: '72px',
                        backgroundColor: result.visible ? '#eff6ff' : '#f1f5f9',
                        color: result.visible ? '#1d4ed8' : '#475569',
                        border: '1px solid #cbd5e1',
                        padding: '4px 12px',
                        borderRadius: '8px',
                        cursor: 'pointer',
                        fontSize: '12px'
                      }}
                    >
                      {result.visible ? '顯示中' : '隱藏'}
                    </button>
                    <button
                      onClick={() => removeResult(result.id)}
                      style={{
                        height: '32px',
                        backgroundColor: '#fff',
                        color: '#be123c',
                        border: '1px solid #fecdd3',
                        padding: '4px 12px',
                        borderRadius: '8px',
                        cursor: 'pointer',
                        fontSize: '12px'
                      }}
                    >
                      移除
                    </button>
                  </div>
                  <div className="training-result-image-actions">
                    <ResultImageCopyButton name={result.name} folderPath={result.folderPath} kind="training" />
                    <ResultImageCopyButton name={result.name} folderPath={result.folderPath} kind="phase" />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {visibleResults.length > 0 && (
            <>
              <h2 style={{ marginTop: '12px', marginBottom: '16px' }}>一般訓練曲線比較</h2>
              <div className="training-plot-grid">
                <FullscreenPlot data={createTraces('trainLoss')} layout={createLayout('Train Loss', 'Loss')} />
                <FullscreenPlot data={createCombinedTraces('valLoss', 'valCopyLoss')} layout={createLayout('Validation Loss (Normal vs Copy)', 'Loss')} />
                <FullscreenPlot data={createCombinedTraces('valAccuracy', 'valCopyAccuracy')} layout={createLayout('Validation Accuracy (Normal vs Copy)', 'Accuracy')} />
                <FullscreenPlot data={createCombinedTraces('valF1', 'valCopyF1')} layout={createLayout('Validation F1 Score (Normal vs Copy)', 'F1')} />
                <FullscreenPlot data={createCombinedTraces('valAuc', 'valCopyAuc')} layout={createLayout('Validation ROC AUC (Normal vs Copy)', 'AUC')} />
              </div>
            </>
          )}

          {visiblePhaseResults.length > 0 && (
            <>
              <h2 style={{ marginTop: '32px', marginBottom: '16px' }}>Phase Analysis 結果比較</h2>
              <div className="training-plot-grid">
                <FullscreenPlot data={createPhaseTraces('accuracy')} layout={createPhaseLayout('Phase Accuracy', 'Accuracy', 1)} />
                <FullscreenPlot data={createPhaseTraces('f1Score')} layout={createPhaseLayout('Phase F1 Score', 'F1', 1)} />
                <FullscreenPlot data={createPhaseTraces('auc')} layout={createPhaseLayout('Phase ROC AUC', 'AUC', 1)} />
                <FullscreenPlot data={createPhaseTraces('avgProbability')} layout={createPhaseLayout('Phase Avg Probability / Confidence', 'Avg Probability', 1)} />
                <FullscreenPlot data={createPhaseTraces('totalSamples')} layout={createPhaseLayout('Phase Sample Count', 'Samples')} />
              </div>
            </>
          )}

          {visibleTurnMetricResults.length > 0 && (
            <>
              <h2 style={{ marginTop: '32px', marginBottom: '16px' }}>Base vs Random Turn Metrics</h2>
              <div className="training-plot-grid">
                <FullscreenPlot data={createTurnMetricTraces('avgProbability')} layout={createLayout('Base vs Random Avg Probability by Turn', 'Avg Probability')} />
                <FullscreenPlot data={createTurnMetricTraces('accuracy')} layout={createLayout('Base vs Random Accuracy by Turn', 'Accuracy')} />
                <FullscreenPlot data={createTurnMetricTraces('f1Score')} layout={createLayout('Base vs Random F1 Score by Turn', 'F1')} />
                <FullscreenPlot data={createTurnMetricTraces('auc')} layout={createLayout('Base vs Random ROC AUC by Turn', 'AUC')} />
                <FullscreenPlot data={createTurnMetricTraces('positiveRate')} layout={createLayout('Base vs Random Positive Rate by Turn', 'Positive Rate')} />
              </div>

              <div style={{ marginTop: '24px' }}>
                <FullscreenPlot data={createBaseRandomTurnMetricTraces()} layout={createBaseRandomTurnMetricsLayout()} style={COMBINED_TURN_PLOT_STYLE} />
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
