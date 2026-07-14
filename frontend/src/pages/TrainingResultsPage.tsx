import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Plot from 'react-plotly.js'

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
  valCopyLoss: (number | null)[]
  valCopyAccuracy: (number | null)[]
  valCopyF1: (number | null)[]
}

type PhaseName = 'early' | 'mid' | 'late'

interface PhaseMetrics {
  phase: PhaseName
  totalSamples: number
  accuracy: number | null
  precision: number | null
  recall: number | null
  f1Score: number | null
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

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        border: '1px solid #e5e5e5',
        borderRadius: isFullscreen ? 0 : '8px',
        padding: isFullscreen ? '12px' : '16px',
        backgroundColor: '#fff',
        boxSizing: 'border-box',
        width: '100%',
        height: isFullscreen ? '100vh' : undefined
      }}
    >
      <button
        type="button"
        onClick={() => void toggleFullscreen()}
        aria-label={isFullscreen ? 'Exit fullscreen chart' : 'View chart fullscreen'}
        title={isFullscreen ? '退出全螢幕' : '全螢幕放大'}
        style={{ position: 'absolute', top: 8, right: 8, zIndex: 10, cursor: 'pointer', padding: '5px 9px' }}
      >
        {isFullscreen ? '縮小' : '全螢幕'}
      </button>
      <Plot
        data={data}
        layout={layout}
        config={PLOT_CONFIG}
        style={isFullscreen ? { width: '100%', height: 'calc(100vh - 24px)' } : style}
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
    valCopyLoss: [],
    valCopyAccuracy: [],
    valCopyF1: []
  }

  let inHistory = false
  let headerIdx = -1

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim()
    if (line === '[TRAINING_HISTORY]') {
      inHistory = true
      headerIdx = i + 1
      continue
    }
    if (inHistory && i === headerIdx) {
      continue
    }
    if (inHistory && line.length > 0) {
      const cols = splitCsvLine(line)
      if (cols.length >= 8) {
        const epoch = parseInt(cols[0], 10)
        const trainLoss = parseFloat(cols[2])
        const valLoss = parseFloat(cols[5])
        const valAcc = parseFloat(cols[6])
        const valF1 = parseFloat(cols[7])
        const valCopyLoss = cols[10] ? parseFloat(cols[10]) : NaN
        const valCopyAcc = cols[11] ? parseFloat(cols[11]) : NaN
        const valCopyF1 = cols[12] ? parseFloat(cols[12]) : NaN
        if (!isNaN(epoch)) {
          data.epochs.push(epoch)
          data.trainLoss.push(isNaN(trainLoss) ? null : trainLoss)
          data.valLoss.push(isNaN(valLoss) ? null : valLoss)
          data.valAccuracy.push(isNaN(valAcc) ? null : valAcc)
          data.valF1.push(isNaN(valF1) ? null : valF1)
          data.valCopyLoss.push(isNaN(valCopyLoss) ? null : valCopyLoss)
          data.valCopyAccuracy.push(isNaN(valCopyAcc) ? null : valCopyAcc)
          data.valCopyF1.push(isNaN(valCopyF1) ? null : valCopyF1)
        }
      }
    }
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

function restoreResults(): TrainingResult[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      const parsed = JSON.parse(stored)
      if (Array.isArray(parsed)) {
        return parsed.map(result => ({
          ...result,
          phaseData: result.phaseData ?? null,
          randomPhaseData: result.randomPhaseData ?? null,
          turnMetrics: result.turnMetrics ?? null,
          randomTurnMetrics: result.randomTurnMetrics ?? null
        }))
      }
    }
  } catch (e) {
    console.error('Failed to restore results from localStorage:', e)
  }
  return []
}

function saveResults(results: TrainingResult[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(results))
  } catch (e) {
    console.error('Failed to save results to localStorage:', e)
  }
}

interface TrainingResultsPageProps {
  onBack: () => void
  requestedFolder?: { path: string; requestId: number } | null
}

export default function TrainingResultsPage({ onBack, requestedFolder }: TrainingResultsPageProps) {
  const [results, setResults] = useState<TrainingResult[]>(() => restoreResults())
  const [folderInput, setFolderInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [scanError, setScanError] = useState<string | null>(null)

  useEffect(() => {
    saveResults(results)
  }, [results])

  const scanFolder = async (folderOverride?: string) => {
    const targetFolder = (folderOverride ?? folderInput).trim()
    if (!targetFolder) return

    setLoading(true)
    setScanError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/api/training/scan`, {
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

      for (const file of data.files) {
        try {
          const readResponse = await fetch(`${API_BASE_URL}/api/training/read_file`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ file_path: file.path }),
          })

          if (!readResponse.ok) {
            console.warn(`Failed to read ${file.name}: HTTP ${readResponse.status}`)
            continue
          }

          const fileData: ReadFileResponse = await readResponse.json()
          if (!fileData.success) {
            console.warn(`Failed to read ${file.name}: ${fileData.error}`)
            continue
          }

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
        setResults(prev => {
          const updated = [...prev, ...newResults]
          saveResults(updated)
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
    if (!requestedFolder?.path) return
    setFolderInput(requestedFolder.path)
    void scanFolder(requestedFolder.path)
    // requestId intentionally permits importing the same folder again.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestedFolder?.requestId])

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
    title: { text: title, font: { size: 16 } },
    xaxis: { title: 'Epoch', gridcolor: '#e5e5e5' },
    yaxis: { title: yaxisTitle, gridcolor: '#e5e5e5' },
    plot_bgcolor: '#fafafa',
    paper_bgcolor: '#ffffff',
    hovermode: 'closest' as const,
    showlegend: true,
    legend: { x: 0, y: 1.1, orientation: 'h' as const }
  })

  const createPhaseLayout = (title: string, yaxisTitle: string, yMax?: number) => ({
    autosize: true,
    title: { text: title, font: { size: 16 } },
    xaxis: { title: 'Phase', gridcolor: '#e5e5e5' },
    yaxis: { title: yaxisTitle, gridcolor: '#e5e5e5', range: yMax ? [0, yMax] : undefined },
    plot_bgcolor: '#fafafa',
    paper_bgcolor: '#ffffff',
    hovermode: 'closest' as const,
    showlegend: true,
    barmode: 'group' as const,
    legend: { x: 0, y: 1.1, orientation: 'h' as const }
  })

  const createBaseRandomTurnMetricsLayout = () => ({
    autosize: true,
    title: { text: 'Turn Metrics Comparison - Base vs RandomFeature', font: { size: 16 }, y: 0.98 },
    xaxis: { title: 'Turn Number (Total Discards)', gridcolor: '#e5e5e5', range: [0, 78] },
    yaxis: { title: 'Probability / Rate', gridcolor: '#e5e5e5', range: [-0.03, 1.03] },
    yaxis2: { title: 'Sample Count', overlaying: 'y' as const, side: 'right' as const, showgrid: false },
    plot_bgcolor: '#fafafa',
    paper_bgcolor: '#ffffff',
    hovermode: 'closest' as const,
    showlegend: true,
    legend: { x: 0, y: 1.18, xanchor: 'left' as const, yanchor: 'bottom' as const, orientation: 'h' as const },
    margin: { l: 60, r: 80, t: 180, b: 80 },
    shapes: [
      { type: 'rect' as const, xref: 'x' as const, yref: 'paper' as const, x0: 0, x1: 24, y0: 0, y1: 1, fillcolor: 'green', opacity: 0.12, line: { width: 0 }, layer: 'below' as const },
      { type: 'rect' as const, xref: 'x' as const, yref: 'paper' as const, x0: 24, x1: 40, y0: 0, y1: 1, fillcolor: 'gold', opacity: 0.12, line: { width: 0 }, layer: 'below' as const },
      { type: 'rect' as const, xref: 'x' as const, yref: 'paper' as const, x0: 40, x1: 78, y0: 0, y1: 1, fillcolor: 'red', opacity: 0.12, line: { width: 0 }, layer: 'below' as const },
      { type: 'line' as const, xref: 'x' as const, yref: 'paper' as const, x0: 24, x1: 24, y0: 0, y1: 1, line: { color: 'orange', width: 2, dash: 'dot' as const } },
      { type: 'line' as const, xref: 'x' as const, yref: 'paper' as const, x0: 40, x1: 40, y0: 0, y1: 1, line: { color: 'purple', width: 2, dash: 'dot' as const } }
    ],
    annotations: [
      { x: 12, y: 1.02, xref: 'x' as const, yref: 'paper' as const, text: 'Early Phase', showarrow: false, font: { size: 11, color: 'green' } },
      { x: 32, y: 1.02, xref: 'x' as const, yref: 'paper' as const, text: 'Mid Phase', showarrow: false, font: { size: 11, color: '#8a6d00' } },
      { x: 59, y: 1.02, xref: 'x' as const, yref: 'paper' as const, text: 'Late Phase', showarrow: false, font: { size: 11, color: 'red' } },
      { x: 24, y: 0.96, xref: 'x' as const, yref: 'paper' as const, text: 'Mid Start (24)', showarrow: false, textangle: -90, font: { size: 10, color: 'orange' } },
      { x: 40, y: 0.96, xref: 'x' as const, yref: 'paper' as const, text: 'Late Start (40)', showarrow: false, textangle: -90, font: { size: 10, color: 'purple' } }
    ]
  })

  return (
    <div style={{ padding: '24px', fontFamily: 'system-ui, sans-serif' }}>
      <h1 style={{ marginTop: 0, marginBottom: '24px' }}>
        <button
          onClick={onBack}
          style={{
            marginRight: '12px',
            padding: '6px 12px',
            backgroundColor: '#f0f0f0',
            border: '1px solid #ccc',
            borderRadius: '4px',
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
            padding: '10px',
            fontSize: '14px',
            border: '1px solid #ccc',
            borderRadius: '4px',
            fontFamily: 'monospace'
          }}
        />
        <button
          onClick={() => void scanFolder()}
          disabled={loading}
          style={{
            padding: '10px 20px',
            backgroundColor: '#4dabf7',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
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
        <div style={{ color: '#888', padding: '40px 0', textAlign: 'center' }}>
          <p>尚無訓練成果</p>
          <p style={{ fontSize: '14px' }}>輸入資料夾路徑後點擊「掃描資料夾」，系統會自動找到訓練曲線與 Phase Analysis CSV 並繪製互動式圖表</p>
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
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  padding: '12px',
                  border: '1px solid #e5e5e5',
                  borderRadius: '8px',
                  backgroundColor: '#fafafa'
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
                  <strong>{result.name}</strong>
                  <div style={{ fontSize: '12px', color: '#888', wordBreak: 'break-all' }}>{result.folderPath}</div>
                  <div style={{ display: 'flex', gap: '6px', marginTop: '4px', fontSize: '11px' }}>
                    {result.data && <span style={{ padding: '2px 6px', borderRadius: '10px', backgroundColor: '#e7f5ff', color: '#1971c2' }}>Training</span>}
                    {(result.phaseData || result.randomPhaseData) && <span style={{ padding: '2px 6px', borderRadius: '10px', backgroundColor: '#ebfbee', color: '#2b8a3e' }}>Phase Analysis</span>}
                    {(result.turnMetrics || result.randomTurnMetrics) && <span style={{ padding: '2px 6px', borderRadius: '10px', backgroundColor: '#fff4e6', color: '#e8590c' }}>Turn Metrics</span>}
                  </div>
                </div>
                <button
                  onClick={() => toggleVisibility(result.id)}
                  style={{
                    backgroundColor: result.visible ? '#28a745' : '#6c757d',
                    color: '#fff',
                    border: 'none',
                    padding: '4px 12px',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '12px'
                  }}
                >
                  {result.visible ? '顯示中' : '隱藏'}
                </button>
                <button
                  onClick={() => removeResult(result.id)}
                  style={{
                    backgroundColor: '#dc3545',
                    color: '#fff',
                    border: 'none',
                    padding: '4px 12px',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    marginLeft: '8px',
                    fontSize: '12px'
                  }}
                >
                  移除
                </button>
              </div>
            ))}
          </div>

          {visibleResults.length > 0 && (
            <>
              <h2 style={{ marginTop: '12px', marginBottom: '16px' }}>一般訓練曲線比較</h2>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                <FullscreenPlot data={createTraces('trainLoss')} layout={createLayout('Train Loss', 'Loss')} />
                <FullscreenPlot data={createCombinedTraces('valLoss', 'valCopyLoss')} layout={createLayout('Validation Loss (Normal vs Copy)', 'Loss')} />
                <FullscreenPlot data={createCombinedTraces('valAccuracy', 'valCopyAccuracy')} layout={createLayout('Validation Accuracy (Normal vs Copy)', 'Accuracy')} />
                <FullscreenPlot data={createCombinedTraces('valF1', 'valCopyF1')} layout={createLayout('Validation F1 Score (Normal vs Copy)', 'F1')} />
              </div>
            </>
          )}

          {visiblePhaseResults.length > 0 && (
            <>
              <h2 style={{ marginTop: '32px', marginBottom: '16px' }}>Phase Analysis 結果比較</h2>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                <FullscreenPlot data={createPhaseTraces('accuracy')} layout={createPhaseLayout('Phase Accuracy', 'Accuracy', 1)} />
                <FullscreenPlot data={createPhaseTraces('f1Score')} layout={createPhaseLayout('Phase F1 Score', 'F1', 1)} />
                <FullscreenPlot data={createPhaseTraces('avgProbability')} layout={createPhaseLayout('Phase Avg Probability / Confidence', 'Avg Probability', 1)} />
                <FullscreenPlot data={createPhaseTraces('totalSamples')} layout={createPhaseLayout('Phase Sample Count', 'Samples')} />
              </div>
            </>
          )}

          {visibleTurnMetricResults.length > 0 && (
            <>
              <h2 style={{ marginTop: '32px', marginBottom: '16px' }}>Base vs Random Turn Metrics</h2>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                <FullscreenPlot data={createTurnMetricTraces('avgProbability')} layout={createLayout('Base vs Random Avg Probability by Turn', 'Avg Probability')} />
                <FullscreenPlot data={createTurnMetricTraces('accuracy')} layout={createLayout('Base vs Random Accuracy by Turn', 'Accuracy')} />
                <FullscreenPlot data={createTurnMetricTraces('f1Score')} layout={createLayout('Base vs Random F1 Score by Turn', 'F1')} />
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
