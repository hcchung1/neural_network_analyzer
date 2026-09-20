import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { copyToClipboard } from '@/lib/clipboard'

interface CsvSession {
  id: string
  sourcePath: string
  displayName: string
  displayPath: string
  headers: string[]
  images: CsvImage[]
  modelPaths: string[]
  trainingFolder: string | null
  pinned: boolean
  currentPage: number
  filterText: string
  rows: string[][]
  startRow: number
  endRow: number
  hasPrevious: boolean
  hasNext: boolean
  totalRows: number | null
  scannedRows: number
  matchedRows: number | null
}

interface CsvImage {
  id: string
  name: string
  url: string
}

interface OpenCsvResponse {
  success: boolean
  session_id: string
  display_name: string
  display_path: string
  headers: string[]
  images: CsvImage[]
  model_paths: string[]
  training_folder: string | null
  error?: string
}

interface CsvPageResponse {
  success: boolean
  headers: string[]
  rows: string[][]
  page: number
  page_size: number
  start_row: number
  end_row: number
  has_previous: boolean
  has_next: boolean
  total_rows: number | null
  scanned_rows: number
  matched_rows: number | null
  error?: string
}

interface CsvNeighbor {
  source_row_number: number
  page: number
  row_index: number
  row: string[]
  tenhou_link: string
}

interface CsvNeighborsResponse {
  success: boolean
  previous: CsvNeighbor | null
  next: CsvNeighbor | null
  error?: string
}

const API_BASE_URL = ''
const PAGE_SIZES = [200, 500, 1000, 2000]
const LINK_COLUMN_NAMES = new Set(['tenhou_link', 'link', 'url'])
const CSV_READER_STORAGE_KEY = 'csv_reader_metadata'
const SIDE_PANE_DEFAULT_WIDTH = 520
const SIDE_PANE_MIN_WIDTH = 280
const SIDE_PANE_MAX_WIDTH = 1200

function clampSidePaneWidth(width: number): number {
  return Math.min(SIDE_PANE_MAX_WIDTH, Math.max(SIDE_PANE_MIN_WIDTH, width))
}

interface PersistedCsvReaderState {
  sessions: Array<{
    sourcePath: string
    currentPage: number
    filterText: string
    pinned?: boolean
  }>
  activeSourcePath: string | null
  pageSize: number
  sidePaneOpen: boolean
  sideMode: 'web' | 'image'
  sideUrl: string
  selectedImageId: string
  imageZoom: number
  sidePaneWidth?: number
  focusMode?: boolean
}

function restorePersistedState(): PersistedCsvReaderState | null {
  try {
    if (typeof window === 'undefined') return null
    const raw = localStorage.getItem(CSV_READER_STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as PersistedCsvReaderState
    if (!parsed || !Array.isArray(parsed.sessions)) return null
    return parsed
  } catch (error) {
    console.error('Failed to restore CSV reader metadata:', error)
    return null
  }
}

function escapeCsvCell(value: string): string {
  if (/[",\n\r]/.test(value)) {
    return `"${value.replace(/"/g, '""')}"`
  }
  return value
}

function rowToCsv(headers: string[], row: string[]): string {
  return headers.map(escapeCsvCell).join(',') + '\n' + headers.map((_, i) => escapeCsvCell(row[i] ?? '')).join(',')
}

function looksLikeUrl(value: string): boolean {
  return /^https?:\/\//i.test(value.trim())
}

function normalizeUrl(value: string): string {
  const trimmed = value.trim()
  if (/^https?:\/\//i.test(trimmed)) return trimmed
  return `https://${trimmed}`
}

function formatNumber(value: number | null | undefined): string {
  return typeof value === 'number' ? value.toLocaleString() : '?'
}

function findHeaderIndex(headers: string[], name: string): number {
  const normalizedName = name.toLowerCase()
  return headers.findIndex(header => header.trim().toLowerCase() === normalizedName)
}

function getCheckpointLabel(path: string): string {
  const fileName = path.split(/[\\/]/).pop() ?? path
  const stem = fileName.replace(/\.(?:pth|pt)$/i, '')
  const epochMatch = stem.match(/(?:^|[_-])epoch[_-]?(\d+)$/i)
  return epochMatch ? `epoch${epochMatch[1]}` : 'all'
}

function compareCheckpointPaths(left: string, right: string): number {
  const leftLabel = getCheckpointLabel(left)
  const rightLabel = getCheckpointLabel(right)
  if (leftLabel === 'all' && rightLabel !== 'all') return -1
  if (rightLabel === 'all' && leftLabel !== 'all') return 1
  if (leftLabel === 'all' && rightLabel === 'all') return left.localeCompare(right)
  const leftEpoch = Number(leftLabel.replace('epoch', ''))
  const rightEpoch = Number(rightLabel.replace('epoch', ''))
  if (leftEpoch !== rightEpoch) return leftEpoch - rightEpoch
  return left.localeCompare(right)
}

interface CsvReaderPageProps {
  onBack: () => void
  onAddModel: (path: string) => void
  onAddTrainingResults: (folder: string) => void
}

interface OutputIndexResponse {
  success: boolean
  root: string
  paths: string[]
  indexed_count: number
  last_scanned_at: number | null
  error?: string
}

export default function CsvReaderPage({ onBack, onAddModel, onAddTrainingResults }: CsvReaderPageProps) {
  const persistedState = useMemo(() => restorePersistedState(), [])
  const [pathInput, setPathInput] = useState('')
  const [isLoaded, setIsLoaded] = useState(false)
  
  useEffect(() => {
    setIsLoaded(true)
  }, [])
  
  const [sessions, setSessions] = useState<CsvSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [pageSize, setPageSize] = useState(persistedState?.pageSize && PAGE_SIZES.includes(persistedState.pageSize) ? persistedState.pageSize : 1000)
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState('No file loaded.')
  const [goToPage, setGoToPage] = useState('1')
  const [selectedCell, setSelectedCell] = useState<{ rowIndex: number; colIndex: number } | null>(null)
  const [selectedRowIndex, setSelectedRowIndex] = useState<number | null>(null)
  const [sidePaneOpen, setSidePaneOpen] = useState(persistedState?.sidePaneOpen ?? false)
  const [sideMode, setSideMode] = useState<'web' | 'image'>(persistedState?.sideMode ?? 'web')
  const [sideUrl, setSideUrl] = useState(persistedState?.sideUrl ?? '')
  const [selectedImageId, setSelectedImageId] = useState(persistedState?.selectedImageId ?? '0')
  const [imageZoom, setImageZoom] = useState(persistedState?.imageZoom ?? 1)
  const [focusMode, setFocusMode] = useState(persistedState?.focusMode ?? false)
  const initialSideWidth = persistedState?.sidePaneWidth
  const [sidePaneWidth, setSidePaneWidth] = useState(() => {
    const w = initialSideWidth ?? SIDE_PANE_DEFAULT_WIDTH
    return clampSidePaneWidth(w)
  })
  const normalSidePaneWidthRef = useRef(sidePaneWidth)
  const splitContainerRef = useRef<HTMLDivElement>(null)
  const tableContainerRef = useRef<HTMLDivElement>(null)
  const sidePaneRef = useRef<HTMLElement>(null)
  const splitSeparatorRef = useRef<HTMLDivElement>(null)
  const [isResizingSplit, setIsResizingSplit] = useState(false)
  const [hasRestoredSessions, setHasRestoredSessions] = useState(false)
  const [isRestoringSessions, setIsRestoringSessions] = useState(false)
  const restoreStartedRef = useRef(false)
  const [libraryOpen, setLibraryOpen] = useState(false)
  const [relatedSessionId, setRelatedSessionId] = useState<string | null>(null)
  const [outputQuery, setOutputQuery] = useState('')
  const [outputMatches, setOutputMatches] = useState<string[]>([])
  const [outputResultsOpen, setOutputResultsOpen] = useState(false)
  const [indexInfo, setIndexInfo] = useState('正在讀取索引狀態…')
  const [indexBusy, setIndexBusy] = useState(false)
  const outputSearchRef = useRef<HTMLDivElement>(null)
  const pendingTableScrollRef = useRef<{ rowIndex: number; colIndex: number } | null>(null)
  const neighborRequestRef = useRef(0)
  const neighborAbortRef = useRef<AbortController | null>(null)
  const navigationRequestRef = useRef(0)
  const navigationAbortRef = useRef<AbortController | null>(null)
  const [predictionNeighbors, setPredictionNeighbors] = useState<{
    previous: CsvNeighbor | null
    next: CsvNeighbor | null
  }>({ previous: null, next: null })
  const [neighborLoading, setNeighborLoading] = useState(false)
  const [navigationBusy, setNavigationBusy] = useState(false)

  const activeSession = useMemo(
    () => sessions.find(session => session.id === activeSessionId) ?? null,
    [sessions, activeSessionId]
  )
  const relatedSession = useMemo(
    () => sessions.find(session => session.id === relatedSessionId) ?? null,
    [relatedSessionId, sessions]
  )

  const updateSession = useCallback((sessionId: string, updater: (session: CsvSession) => CsvSession) => {
    setSessions(prev => prev.map(session => session.id === sessionId ? updater(session) : session))
  }, [])

  const loadPage = useCallback(async (sessionId: string, page: number, filterText?: string, includeTotal = false, signal?: AbortSignal): Promise<boolean> => {
    const session = sessions.find(s => s.id === sessionId)
    if (!session) return false

    setLoading(true)
    setStatus(filterText ?? session.filterText ? `Loading filtered page ${page + 1}...` : `Loading page ${page + 1}...`)
    try {
      const response = await fetch(`${API_BASE_URL}/api/legacy/csv_reader/page`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal,
        body: JSON.stringify({
          session_id: sessionId,
          page,
          page_size: pageSize,
          filter_text: filterText ?? session.filterText,
          include_total: includeTotal
        })
      })
      const data: CsvPageResponse = await response.json()
      if (!data.success) {
        setStatus(`Load failed: ${data.error || 'unknown error'}`)
        return false
      }
      updateSession(sessionId, old => ({
        ...old,
        headers: data.headers,
        rows: data.rows,
        currentPage: data.page,
        startRow: data.start_row,
        endRow: data.end_row,
        hasPrevious: data.has_previous,
        hasNext: data.has_next,
        totalRows: data.total_rows,
        scannedRows: data.scanned_rows,
        matchedRows: data.matched_rows,
        filterText: filterText ?? old.filterText
      }))
      setGoToPage(String(data.page + 1))
      setSelectedCell(null)
      setSelectedRowIndex(null)
      const totalText = data.total_rows === null ? 'total indexing in progress' : `${data.total_rows.toLocaleString()} total`
      setStatus(`Rows ${formatNumber(data.start_row)}-${formatNumber(data.end_row)} (${totalText}; scanned ${data.scanned_rows.toLocaleString()})`)
      return true
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') return false
      setStatus(`Load failed: ${error instanceof Error ? error.message : String(error)}`)
      return false
    } finally {
      setLoading(false)
    }
  }, [pageSize, sessions, updateSession])

  const openCsvPath = useCallback(async (path: string, restorePage = 0, restoreFilterText = '', pinned = true): Promise<string | null> => {
    const normalizedPath = path.trim()
    if (!normalizedPath) return null
    setLoading(true)
    setStatus(`Opening ${normalizedPath}...`)
    try {
      const response = await fetch(`${API_BASE_URL}/api/legacy/csv_reader/open`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: normalizedPath })
      })
      const data: OpenCsvResponse = await response.json()
      if (!data.success) {
        setStatus(`Open failed: ${data.error || 'unknown error'}`)
        return null
      }
      const newSession: CsvSession = {
        id: data.session_id,
        sourcePath: normalizedPath,
        displayName: data.display_name,
        displayPath: data.display_path,
        headers: data.headers,
        images: data.images || [],
        modelPaths: data.model_paths || [],
        trainingFolder: data.training_folder ?? null,
        pinned,
        currentPage: Math.max(0, restorePage),
        filterText: restoreFilterText,
        rows: [],
        startRow: 0,
        endRow: 0,
        hasPrevious: false,
        hasNext: false,
        totalRows: null,
        scannedRows: 0,
        matchedRows: null
      }
      setSessions(prev => [...prev, newSession])
      setActiveSessionId(data.session_id)
      setSelectedImageId(current => data.images?.[0]?.id ?? current)
      if (data.images?.length) {
        setSidePaneOpen(true)
        setSideMode('image')
      }
      setStatus(`Opened ${data.display_name}`)
      return data.session_id
    } catch (error) {
      setStatus(`Open failed: ${error instanceof Error ? error.message : String(error)}`)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  const openPath = async () => {
    const opened = await openCsvPath(pathInput)
    if (opened) setPathInput('')
  }

  useEffect(() => {
    if (restoreStartedRef.current) return
    restoreStartedRef.current = true

    if (!persistedState?.sessions.length) {
      setHasRestoredSessions(true)
      return
    }

    const restoreSessions = async () => {
      setIsRestoringSessions(true)
      setStatus(`Restoring ${persistedState.sessions.length} CSV session(s)...`)
      const restored: Array<{ sourcePath: string; sessionId: string }> = []
      try {
        for (const saved of persistedState.sessions) {
          const sessionId = await openCsvPath(saved.sourcePath, saved.currentPage, saved.filterText, saved.pinned ?? true)
          if (sessionId) {
            restored.push({ sourcePath: saved.sourcePath, sessionId })
          }
        }
        const active = restored.find(item => item.sourcePath === persistedState.activeSourcePath) ?? restored[restored.length - 1]
        const activeSaved = persistedState.sessions.find(item => item.sourcePath === active?.sourcePath)
        if (active) {
          setActiveSessionId(active.sessionId)
          if (activeSaved) {
            setGoToPage(String(Math.max(0, activeSaved.currentPage) + 1))
          }
        }
        setStatus(restored.length > 0 ? `Restored ${restored.length} CSV session(s).` : 'No CSV session restored.')
      } finally {
        setIsRestoringSessions(false)
        setHasRestoredSessions(true)
      }
    }

    void restoreSessions()
  }, [openCsvPath, persistedState])

  const saveState = useCallback(() => {
    const payload: PersistedCsvReaderState = {
      sessions: sessions.map(session => ({
        sourcePath: session.sourcePath,
        currentPage: session.currentPage,
        filterText: session.filterText,
        pinned: session.pinned
      })),
      activeSourcePath: activeSession?.sourcePath ?? null,
      pageSize,
      sidePaneOpen,
      sideMode,
      sideUrl,
      selectedImageId,
      imageZoom,
      sidePaneWidth: focusMode ? normalSidePaneWidthRef.current : sidePaneWidth,
      focusMode
    }
    try {
      if (typeof window === 'undefined') return
      localStorage.setItem(CSV_READER_STORAGE_KEY, JSON.stringify(payload))
    } catch (error) {
      console.error('Failed to save CSV reader metadata:', error)
    }
  }, [activeSession?.sourcePath, focusMode, imageZoom, pageSize, selectedImageId, sessions, sideMode, sidePaneOpen, sidePaneWidth, sideUrl])

  // Save on meaningful state transitions, NOT every render
  useEffect(() => {
    if (!hasRestoredSessions || isRestoringSessions) return
    saveState()
  }, [saveState, hasRestoredSessions, isRestoringSessions])

  const resizeRafRef = useRef<number | null>(null)
  const resizePointerIdRef = useRef<number | null>(null)
  const resizeContainerRightRef = useRef(0)
  const resizePendingClientXRef = useRef<number | null>(null)
  const resizeWidthRef = useRef(sidePaneWidth)

  const applySidePaneRatio = useCallback((ratio: number) => {
    const container = splitContainerRef.current
    if (!container || !sidePaneOpen) return
    const width = clampSidePaneWidth(Math.round(container.clientWidth * ratio))
    resizeWidthRef.current = width
    setSidePaneWidth(width)
  }, [sidePaneOpen])

  const enterFocusMode = useCallback(() => {
    normalSidePaneWidthRef.current = sidePaneWidth
    setFocusMode(true)
  }, [sidePaneWidth])

  const exitFocusMode = useCallback(() => {
    setSidePaneWidth(normalSidePaneWidthRef.current)
    resizeWidthRef.current = normalSidePaneWidthRef.current
    setFocusMode(false)
  }, [])

  const applyPendingResize = useCallback(() => {
    resizeRafRef.current = null
    const clientX = resizePendingClientXRef.current
    if (clientX === null) return
    resizePendingClientXRef.current = null

    const width = clampSidePaneWidth(resizeContainerRightRef.current - clientX)
    resizeWidthRef.current = width
    if (sidePaneRef.current) sidePaneRef.current.style.width = `${width}px`
    splitSeparatorRef.current?.setAttribute('aria-valuenow', String(Math.round(width)))
  }, [])

  const queueResize = useCallback((clientX: number) => {
    resizePendingClientXRef.current = clientX
    if (resizeRafRef.current === null) {
      resizeRafRef.current = requestAnimationFrame(applyPendingResize)
    }
  }, [applyPendingResize])

  const finishResize = useCallback((pointerId?: number) => {
    const activePointerId = resizePointerIdRef.current
    if (activePointerId === null || (pointerId !== undefined && pointerId !== activePointerId)) return

    if (resizeRafRef.current !== null) {
      cancelAnimationFrame(resizeRafRef.current)
      resizeRafRef.current = null
    }
    applyPendingResize()
    resizePointerIdRef.current = null

    const separator = splitSeparatorRef.current
    if (separator?.hasPointerCapture(activePointerId)) {
      separator.releasePointerCapture(activePointerId)
    }

    setIsResizingSplit(false)
    setSidePaneWidth(resizeWidthRef.current)
  }, [applyPendingResize])

  const handleResizePointerDown = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || resizePointerIdRef.current !== null) return
    const container = splitContainerRef.current
    if (!container) return

    event.preventDefault()
    resizePointerIdRef.current = event.pointerId
    resizeContainerRightRef.current = container.getBoundingClientRect().right
    resizePendingClientXRef.current = event.clientX
    resizeWidthRef.current = sidePaneWidth
    event.currentTarget.setPointerCapture(event.pointerId)
    setIsResizingSplit(true)
  }, [sidePaneWidth])

  const handleResizePointerMove = useCallback((event: React.PointerEvent) => {
    if (event.pointerId !== resizePointerIdRef.current) return
    queueResize(event.clientX)
  }, [queueResize])

  const handleResizePointerEnd = useCallback((event: React.PointerEvent) => {
    finishResize(event.pointerId)
  }, [finishResize])

  useEffect(() => {
    if (!isResizingSplit) return
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    const onWindowBlur = () => finishResize()
    window.addEventListener('blur', onWindowBlur)
    return () => {
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      window.removeEventListener('blur', onWindowBlur)
    }
  }, [finishResize, isResizingSplit])

  useEffect(() => () => {
    if (resizeRafRef.current !== null) cancelAnimationFrame(resizeRafRef.current)
  }, [])

  useEffect(() => {
    if (!focusMode) return
    const exitOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') exitFocusMode()
    }
    window.addEventListener('keydown', exitOnEscape)
    return () => window.removeEventListener('keydown', exitOnEscape)
  }, [exitFocusMode, focusMode])

  useEffect(() => {
    if (activeSession && activeSession.rows.length === 0) {
      loadPage(activeSession.id, activeSession.currentPage)
    }
  }, [activeSession, loadPage])

  const closeSession = useCallback(async (sessionId: string) => {
    const closingIndex = sessions.findIndex(session => session.id === sessionId)
    if (closingIndex === -1) return

    const closingSession = sessions[closingIndex]
    const remaining = sessions.filter(session => session.id !== sessionId)
    const isClosingActiveSession = activeSessionId === sessionId

    setSessions(remaining)
    if (relatedSessionId === sessionId) setRelatedSessionId(null)
    if (isClosingActiveSession) {
      const nextSession = remaining[Math.min(closingIndex, remaining.length - 1)] ?? null
      setActiveSessionId(nextSession?.id ?? null)
      setGoToPage(String((nextSession?.currentPage ?? 0) + 1))
      setSelectedCell(null)
      setSelectedRowIndex(null)
    }
    setStatus(`Closed ${closingSession.displayName}.`)

    try {
      const response = await fetch(`${API_BASE_URL}/api/legacy/csv_reader/close`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
    } catch (error) {
      setStatus(`Closed ${closingSession.displayName} locally; backend cleanup failed: ${error instanceof Error ? error.message : String(error)}`)
    }
  }, [activeSessionId, relatedSessionId, sessions])

  const applyFilter = () => {
    if (!activeSession) return
    loadPage(activeSession.id, 0, activeSession.filterText, true)
  }

  const clearFilter = () => {
    if (!activeSession) return
    updateSession(activeSession.id, old => ({ ...old, filterText: '' }))
    loadPage(activeSession.id, 0, '', true)
  }

  const copyText = async (text: string, message: string) => {
    try {
      await copyToClipboard(text)
      setStatus(message)
    } catch (error) {
      setStatus(`Copy failed: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  const copySelectedCell = () => {
    if (!activeSession || !selectedCell) return
    copyText(activeSession.rows[selectedCell.rowIndex]?.[selectedCell.colIndex] ?? '', 'Copied cell value.')
  }

  const copySelectedRow = () => {
    if (!activeSession || selectedRowIndex === null) return
    const row = activeSession.rows[selectedRowIndex]
    if (!row) return
    copyText(rowToCsv(activeSession.headers, row), 'Copied row as CSV.')
  }

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'c' && activeSession && selectedRowIndex !== null) {
        event.preventDefault()
        copySelectedRow()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  })

  const openLinkExternal = (value: string) => {
    const url = normalizeUrl(value)
    window.open(url, '_blank', 'noopener,noreferrer')
    setStatus('Opened link in browser.')
  }

  const loadPredictionNeighbors = useCallback(async (session: CsvSession, row: string[]) => {
    const requestId = ++neighborRequestRef.current
    neighborAbortRef.current?.abort()
    neighborAbortRef.current = null
    setPredictionNeighbors({ previous: null, next: null })
    setNeighborLoading(false)

    const idsIndex = findHeaderIndex(session.headers, 'ids')
    const lineNumberIndex = findHeaderIndex(session.headers, 'line_number')
    if (idsIndex < 0 || lineNumberIndex < 0) {
      setStatus('無法導覽：CSV 需要 ids 與 line_number 欄位。')
      return
    }

    const ids = row[idsIndex]?.trim() ?? ''
    const lineNumber = row[lineNumberIndex]?.trim() ?? ''
    if (!ids || !lineNumber) {
      setStatus('無法導覽：目前列缺少 ids 或 line_number。')
      return
    }

    const controller = new AbortController()
    neighborAbortRef.current = controller
    setNeighborLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/legacy/csv_reader/neighbors`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          session_id: session.id,
          line_number: lineNumber,
          ids,
          page_size: pageSize
        })
      })
      const data: CsvNeighborsResponse = await response.json()
      if (requestId !== neighborRequestRef.current) return
      if (!data.success) {
        setPredictionNeighbors({ previous: null, next: null })
        setStatus(`無法讀取相鄰預測：${data.error || 'unknown error'}`)
        return
      }
      setPredictionNeighbors({ previous: data.previous, next: data.next })
      if (!data.previous && !data.next) {
        setStatus('整份 CSV 中沒有其他相同觀察者與預測對象的紀錄。')
      }
    } catch (error) {
      if (requestId !== neighborRequestRef.current) return
      if (error instanceof Error && error.name === 'AbortError') return
      setPredictionNeighbors({ previous: null, next: null })
      setStatus(`無法讀取相鄰預測：${error instanceof Error ? error.message : String(error)}`)
    } finally {
      if (requestId === neighborRequestRef.current) {
        if (neighborAbortRef.current === controller) neighborAbortRef.current = null
        setNeighborLoading(false)
      }
    }
  }, [pageSize])

  const openLinkInPane = (value: string, row: string[]) => {
    navigationAbortRef.current?.abort()
    navigationAbortRef.current = null
    navigationRequestRef.current += 1
    pendingTableScrollRef.current = null
    setNavigationBusy(false)
    setSideUrl(normalizeUrl(value))
    setSideMode('web')
    setSidePaneOpen(true)
    setStatus('Opened link in side pane.')
    if (activeSession) void loadPredictionNeighbors(activeSession, row)
  }

  const navigateToPrediction = async (direction: 'previous' | 'next') => {
    if (!activeSession || neighborLoading || navigationBusy || loading) return
    const target = predictionNeighbors[direction]
    if (!target?.tenhou_link) return

    const requestId = ++navigationRequestRef.current
    navigationAbortRef.current?.abort()
    const controller = new AbortController()
    navigationAbortRef.current = controller
    const linkColumnIndex = findHeaderIndex(activeSession.headers, 'tenhou_link')
    const targetPage = Math.floor((target.source_row_number - 1) / pageSize)
    const targetRowIndex = (target.source_row_number - 1) % pageSize
    setNavigationBusy(true)
    pendingTableScrollRef.current = {
      rowIndex: targetRowIndex,
      colIndex: Math.max(0, linkColumnIndex)
    }
    try {
      const loaded = await loadPage(activeSession.id, targetPage, '', false, controller.signal)
      if (requestId !== navigationRequestRef.current) return
      if (!loaded) {
        pendingTableScrollRef.current = null
        return
      }
      setSelectedRowIndex(targetRowIndex)
      if (linkColumnIndex >= 0) {
        setSelectedCell({ rowIndex: targetRowIndex, colIndex: linkColumnIndex })
      }
      setSideUrl(normalizeUrl(target.tenhou_link))
      setSideMode('web')
      setSidePaneOpen(true)
      setStatus(`已前往第 ${target.source_row_number.toLocaleString()} 列。`)
      void loadPredictionNeighbors(activeSession, target.row)
    } finally {
      if (requestId === navigationRequestRef.current) {
        if (navigationAbortRef.current === controller) navigationAbortRef.current = null
        setNavigationBusy(false)
      }
    }
  }

  useEffect(() => {
    neighborAbortRef.current?.abort()
    neighborAbortRef.current = null
    navigationAbortRef.current?.abort()
    navigationAbortRef.current = null
    setPredictionNeighbors({ previous: null, next: null })
    neighborRequestRef.current += 1
    setNeighborLoading(false)
    navigationRequestRef.current += 1
    pendingTableScrollRef.current = null
    setNavigationBusy(false)
  }, [activeSessionId])

  useEffect(() => () => {
    neighborAbortRef.current?.abort()
    navigationAbortRef.current?.abort()
  }, [])

  useEffect(() => {
    const pending = pendingTableScrollRef.current
    if (!pending || !tableContainerRef.current) return
    const cell = tableContainerRef.current.querySelector<HTMLElement>(
      `[data-csv-row="${pending.rowIndex}"][data-csv-column="${pending.colIndex}"]`
    )
    if (!cell) return
    pendingTableScrollRef.current = null
    cell.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'center' })
  }, [activeSession?.rows, selectedCell, selectedRowIndex])

  const togglePinned = (sessionId: string) => {
    setSessions(prev => prev.map(session => session.id === sessionId ? { ...session, pinned: !session.pinned } : session))
  }

  const formatIndexInfo = useCallback((data: OutputIndexResponse) => data.last_scanned_at
    ? `${data.indexed_count} 個檔案 · ${new Date(data.last_scanned_at * 1000).toLocaleString()}`
    : '索引尚未建立，請按「重新掃描索引」', [])

  useEffect(() => {
    let cancelled = false
    const loadIndexStatus = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/legacy/csv_reader/output_index/status`)
        const data: OutputIndexResponse = await response.json()
        if (!data.success) throw new Error(data.error || '無法讀取索引狀態')
        if (!cancelled) setIndexInfo(formatIndexInfo(data))
      } catch (error) {
        if (!cancelled) setIndexInfo(error instanceof Error ? error.message : String(error))
      }
    }
    void loadIndexStatus()
    return () => { cancelled = true }
  }, [formatIndexInfo])

  useEffect(() => {
    if (!outputResultsOpen) return
    const closeOnOutsidePointer = (event: PointerEvent) => {
      if (!outputSearchRef.current?.contains(event.target as Node)) setOutputResultsOpen(false)
    }
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOutputResultsOpen(false)
    }
    document.addEventListener('pointerdown', closeOnOutsidePointer)
    window.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('pointerdown', closeOnOutsidePointer)
      window.removeEventListener('keydown', closeOnEscape)
    }
  }, [outputResultsOpen])

  const searchOutputIndex = useCallback(async (query = outputQuery) => {
    setIndexBusy(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/legacy/csv_reader/output_index/search?query=${encodeURIComponent(query)}&limit=40`)
      const data: OutputIndexResponse = await response.json()
      if (!data.success) throw new Error(data.error || '搜尋失敗')
      setOutputMatches(data.paths || [])
      setOutputResultsOpen((data.paths || []).length > 0)
      setIndexInfo(formatIndexInfo(data))
    } catch (error) {
      setOutputResultsOpen(false)
      setIndexInfo(error instanceof Error ? error.message : String(error))
    } finally {
      setIndexBusy(false)
    }
  }, [formatIndexInfo, outputQuery])

  const rebuildOutputIndex = async () => {
    setIndexBusy(true)
    setIndexInfo('正在掃描 Transformer/output...')
    try {
      const response = await fetch(`${API_BASE_URL}/api/legacy/csv_reader/output_index/scan`, { method: 'POST' })
      const data: OutputIndexResponse = await response.json()
      if (!data.success) throw new Error(data.error || '掃描失敗')
      setIndexInfo(formatIndexInfo(data))
      await searchOutputIndex()
    } catch (error) {
      setIndexInfo(error instanceof Error ? error.message : String(error))
    } finally {
      setIndexBusy(false)
    }
  }

  const selectedImage = activeSession?.images.find(img => img.id === selectedImageId) ?? activeSession?.images[0]

  if (!isLoaded) return null;
  return (
    <div style={{
      padding: focusMode ? '8px' : '16px',
      fontFamily: 'system-ui, sans-serif',
      height: focusMode ? '100vh' : '100%',
      boxSizing: 'border-box',
      display: 'flex',
      flexDirection: 'column',
      ...(focusMode ? { position: 'fixed' as const, inset: 0, zIndex: 2000, backgroundColor: '#f8fafc' } : {})
    }}>
      {focusMode ? (
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center', minHeight: '36px', marginBottom: '6px' }}>
          <button onClick={exitFocusMode} style={{ padding: '6px 10px' }}>退出專注模式</button>
          <strong title={activeSession?.displayName} style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '13px' }}>
            {activeSession?.displayName ?? 'Tenhou CSV Reader'}
          </strong>
          <button onClick={() => applySidePaneRatio(0.5)} disabled={!sidePaneOpen}>左右各半</button>
          <button onClick={() => applySidePaneRatio(0.6)} disabled={!sidePaneOpen}>牌面 60%</button>
          <small style={{ color: '#6b7280' }}>Esc 離開</small>
        </div>
      ) : (
        <>
          <h1 style={{ marginTop: 0, marginBottom: '16px', display: 'flex', alignItems: 'center' }}>
            <button onClick={onBack} style={{ marginRight: '12px', padding: '6px 12px' }}>Back</button>
            <span style={{ flex: 1 }}>Tenhou CSV Reader</span>
            <button onClick={enterFocusMode} disabled={!activeSession} style={{ padding: '6px 12px', fontSize: '13px' }}>專注模式</button>
          </h1>

          <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap', marginBottom: '10px' }}>
            <input
              value={pathInput}
              onChange={e => setPathInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') openPath() }}
              placeholder="輸入 server 上的 .csv 或 .zip 路徑"
              style={{ flex: '1 1 520px', padding: '8px', fontFamily: 'monospace' }}
            />
            <button onClick={openPath} disabled={loading} style={{ padding: '8px 14px' }}>Open CSV/ZIP</button>
            <label>Page Size: </label>
            <select value={pageSize} onChange={e => setPageSize(Number(e.target.value))} disabled={loading}>
              {PAGE_SIZES.map(size => <option key={size} value={size}>{size}</option>)}
            </select>
          </div>

          <div className="csv-output-search" ref={outputSearchRef}>
            <input
              value={outputQuery}
              onChange={e => { setOutputQuery(e.target.value); setOutputResultsOpen(false) }}
              onKeyDown={e => {
                if (e.key === 'Enter') void searchOutputIndex()
                if (e.key === 'Escape') setOutputResultsOpen(false)
              }}
              placeholder="搜尋 Transformer/output 內的資料夾或檔名"
            />
            <button onClick={() => void searchOutputIndex()} disabled={indexBusy}>搜尋</button>
            <button onClick={() => void rebuildOutputIndex()} disabled={indexBusy}>重新掃描索引</button>
            <span>{indexInfo}</span>
            {outputResultsOpen && outputMatches.length > 0 && <div className="csv-output-matches" role="listbox">
              {outputMatches.map(path => <button key={path} role="option" title={path} onClick={() => { setPathInput(path); setOutputMatches([]); setOutputResultsOpen(false) }}>{path}</button>)}
            </div>}
          </div>

          <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap', marginBottom: '10px' }}>
            <label>Filter:</label>
            <input
              value={activeSession?.filterText ?? ''}
              onChange={e => activeSession && updateSession(activeSession.id, old => ({ ...old, filterText: e.target.value }))}
              onKeyDown={e => { if (e.key === 'Enter') applyFilter() }}
              placeholder="prediction = 1, actual = 0, probability > 0.7"
              disabled={!activeSession || loading}
              style={{ flex: '1 1 520px', padding: '6px' }}
            />
            <button onClick={applyFilter} disabled={!activeSession || loading}>Apply</button>
            <button onClick={clearFilter} disabled={!activeSession || loading}>Clear</button>
          </div>
        </>
      )}

      <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap', marginBottom: '10px' }}>
        <button onClick={() => activeSession && loadPage(activeSession.id, activeSession.currentPage - 1)} disabled={!activeSession?.hasPrevious || loading}>Previous</button>
        <button onClick={() => activeSession && loadPage(activeSession.id, activeSession.currentPage + 1)} disabled={!activeSession?.hasNext || loading}>Next</button>
        <label>Go to page:</label>
        <input value={goToPage} onChange={e => setGoToPage(e.target.value)} style={{ width: '80px', padding: '4px' }} />
        <button onClick={() => activeSession && loadPage(activeSession.id, Math.max(0, Number(goToPage) - 1))} disabled={!activeSession || loading}>Go</button>
        <span style={{ color: '#586069', marginLeft: '8px' }}>{status}</span>
      </div>

      {!focusMode && sessions.length > 0 && <div className="csv-session-toolbar">
        <button className="csv-library-toggle" onClick={() => setLibraryOpen(open => !open)} aria-expanded={libraryOpen}>
          <span>檔案庫</span><span className={libraryOpen ? 'chevron open' : 'chevron'}>›</span><small>{sessions.length}</small>
        </button>
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', flex: 1 }}>
          {sessions.filter(session => session.pinned).map(session => {
            const isActive = session.id === activeSessionId
            return (
              <div
                key={session.id}
                style={{
                  display: 'inline-flex',
                  alignItems: 'stretch',
                  border: isActive ? '2px solid #228be6' : '1px solid #ccc',
                  borderRadius: '4px',
                  backgroundColor: isActive ? '#e7f5ff' : '#fff',
                  overflow: 'hidden'
                }}
              >
                <button
                  type="button"
                  onClick={() => setActiveSessionId(session.id)}
                  title={session.displayPath}
                  aria-pressed={isActive}
                  style={{
                    padding: '6px 8px 6px 10px',
                    border: 0,
                    backgroundColor: 'transparent',
                    cursor: 'pointer',
                    maxWidth: '320px',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap'
                  }}
                >
                  {session.displayName}
                </button>
                {(session.modelPaths.length > 0 || session.trainingFolder) && <button
                  type="button"
                  className="csv-related-trigger"
                  onClick={() => {
                    setActiveSessionId(session.id)
                    setLibraryOpen(false)
                    setRelatedSessionId(session.id)
                  }}
                  aria-label={`Open related results for ${session.displayName}`}
                  title="查看相關模型與訓練成果"
                >
                  ◈
                </button>}
                <button
                  type="button"
                  onClick={() => void closeSession(session.id)}
                  disabled={loading}
                  aria-label={`Close ${session.displayName}`}
                  title={`Close ${session.displayName}`}
                  style={{
                    width: '28px',
                    flex: '0 0 28px',
                    padding: 0,
                    border: 0,
                    borderLeft: '1px solid #d0d7de',
                    backgroundColor: 'transparent',
                    color: '#495057',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    fontSize: '17px',
                    lineHeight: 1
                  }}
                >
                  ×
                </button>
              </div>
            )
          })}
        </div>
      </div>}

      <aside className={libraryOpen ? 'csv-library open' : 'csv-library'} aria-hidden={!libraryOpen}>
        <div className="csv-library-backdrop" onClick={() => setLibraryOpen(false)} />
        <div className="csv-library-panel">
          <header><div><strong>已開啟檔案</strong><small>控制橫幅顯示，不會關閉 session</small></div><button onClick={() => setLibraryOpen(false)}>×</button></header>
          <div className="csv-library-list">{sessions.map(session => <div key={session.id} className={session.id === activeSessionId ? 'csv-library-item active' : 'csv-library-item'}>
            <button className="csv-library-name" onClick={() => { setActiveSessionId(session.id); setLibraryOpen(false) }} title={session.displayPath}>{session.displayName}<small>{session.sourcePath}</small></button>
            <label className="modern-switch" title="顯示於快速切換橫幅"><input type="checkbox" checked={session.pinned} onChange={() => togglePinned(session.id)} /><span /></label>
          </div>)}</div>
        </div>
      </aside>

      <aside className={relatedSession ? 'csv-related-drawer open' : 'csv-related-drawer'} aria-hidden={!relatedSession}>
        <div className="csv-related-backdrop" onClick={() => setRelatedSessionId(null)} />
        <div className="csv-related-panel">
          <header>
            <div><strong>相關模型與訓練成果</strong><small>{relatedSession?.displayName}</small></div>
            <button onClick={() => setRelatedSessionId(null)} aria-label="Close related results">×</button>
          </header>
          {relatedSession && <div className="csv-related-content">
            {relatedSession.trainingFolder && <section>
              <h3>Training Results</h3>
              <button className="csv-training-result-button" onClick={() => onAddTrainingResults(relatedSession.trainingFolder!)} title={relatedSession.trainingFolder}>
                <span>加入 Training Results</span><strong>{relatedSession.trainingFolder.split('/').pop()}</strong><small>{relatedSession.trainingFolder}</small>
              </button>
            </section>}
            {relatedSession.modelPaths.length > 0 && <section>
              <h3>Transformer Models <span>{relatedSession.modelPaths.length}</span></h3>
              <div className="csv-related-models">{[...relatedSession.modelPaths].sort(compareCheckpointPaths).map(path => <button key={path} onClick={() => onAddModel(path)} title={path}>
                <span>加入 Token Visualization</span><strong className="csv-checkpoint-label">{getCheckpointLabel(path)}</strong><small>{path.split(/[\\/]/).pop()}</small>
              </button>)}</div>
            </section>}
          </div>}
        </div>
      </aside>

      <div
        ref={splitContainerRef}
        style={{
          display: 'flex',
          flexDirection: 'row',
          gap: 0,
          minHeight: 0,
          flex: 1,
          alignItems: 'stretch'
        }}
      >
        <div ref={tableContainerRef} style={{ flex: 1, minWidth: 0, overflow: 'auto', border: '1px solid #d0d7de', borderRadius: '6px', backgroundColor: '#fff' }}>
          {!activeSession ? (
            <div style={{ padding: '40px', textAlign: 'center', color: '#888' }}>Open a CSV or ZIP file to begin.</div>
          ) : (
            <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: '12px' }}>
              <thead style={{ position: 'sticky', top: 0, backgroundColor: '#f6f8fa', zIndex: 1 }}>
                <tr>
                  {activeSession.headers.map(header => (
                    <th key={header} style={{ borderBottom: '1px solid #d0d7de', borderRight: '1px solid #eee', padding: '6px', textAlign: 'left', whiteSpace: 'nowrap' }}>{header}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {activeSession.rows.map((row, rowIndex) => (
                  <tr
                    key={rowIndex}
                    onClick={() => setSelectedRowIndex(rowIndex)}
                    style={{ backgroundColor: selectedRowIndex === rowIndex ? '#e7f5ff' : undefined }}
                  >
                    {activeSession.headers.map((header, colIndex) => {
                      const value = row[colIndex] ?? ''
                      const isLink = LINK_COLUMN_NAMES.has(header.toLowerCase()) || looksLikeUrl(value)
                      return (
                        <td
                          key={`${rowIndex}-${colIndex}`}
                          data-csv-row={rowIndex}
                          data-csv-column={colIndex}
                          onClick={() => setSelectedCell({ rowIndex, colIndex })}
                          onDoubleClick={() => isLink && openLinkInPane(value, row)}
                          onMouseDown={event => {
                            if (event.ctrlKey && isLink) {
                              event.preventDefault()
                              openLinkExternal(value)
                            }
                          }}
                          style={{
                            borderBottom: '1px solid #eee',
                            borderRight: '1px solid #f1f3f5',
                            padding: '5px 6px',
                            maxWidth: header.toLowerCase() === 'tenhou_link' ? '420px' : '260px',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            color: isLink ? '#1c7ed6' : undefined,
                            textDecoration: isLink ? 'underline' : undefined,
                            cursor: isLink ? 'pointer' : 'default',
                            outline: selectedCell?.rowIndex === rowIndex && selectedCell?.colIndex === colIndex ? '2px solid #4dabf7' : undefined
                          }}
                          title={value}
                        >
                          {value}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {sidePaneOpen && (
          <>
            {isResizingSplit && (
              <div
                aria-hidden="true"
                onPointerMove={handleResizePointerMove}
                onPointerUp={handleResizePointerEnd}
                onPointerCancel={handleResizePointerEnd}
                style={{
                  position: 'fixed',
                  inset: 0,
                  zIndex: 9999,
                  cursor: 'col-resize',
                  touchAction: 'none'
                }}
              />
            )}
            <div
              ref={splitSeparatorRef}
              role="separator"
              aria-orientation="vertical"
              aria-label="Resize side panel"
              aria-valuemin={SIDE_PANE_MIN_WIDTH}
              aria-valuemax={SIDE_PANE_MAX_WIDTH}
              aria-valuenow={Math.round(sidePaneWidth)}
              onPointerDown={handleResizePointerDown}
              onPointerMove={handleResizePointerMove}
              onPointerUp={handleResizePointerEnd}
              onPointerCancel={handleResizePointerEnd}
              onLostPointerCapture={handleResizePointerEnd}
              style={{
                width: '8px',
                flexShrink: 0,
                cursor: 'col-resize',
                touchAction: 'none',
                backgroundColor: isResizingSplit ? '#228be6' : '#e9ecef',
                borderLeft: '1px solid #d0d7de',
                borderRight: '1px solid #d0d7de',
                margin: '0 2px',
                borderRadius: '2px'
              }}
              title="Drag to resize"
            />
            <aside
              ref={sidePaneRef}
              style={{
                width: sidePaneWidth,
                flexShrink: 0,
                border: '1px solid #d0d7de',
                borderRadius: '6px',
                backgroundColor: '#fff',
                minHeight: 0,
                display: 'flex',
                flexDirection: 'column'
              }}
            >
            <div style={{ padding: '8px', borderBottom: '1px solid #d0d7de', display: 'flex', gap: '6px', alignItems: 'center', flexWrap: 'wrap' }}>
              <button onClick={() => setSideMode(sideMode === 'web' ? 'image' : 'web')} disabled={!activeSession?.images.length}>{sideMode === 'web' ? 'Show Plot' : 'Show Web'}</button>
              {sideMode === 'web' ? (
                <>
                  <button
                    onClick={() => void navigateToPrediction('previous')}
                    disabled={loading || navigationBusy || neighborLoading || !predictionNeighbors.previous?.tenhou_link}
                    title={predictionNeighbors.previous?.tenhou_link ? `前往 CSV 第 ${predictionNeighbors.previous.source_row_number} 列` : predictionNeighbors.previous ? '上一筆預測缺少 tenhou_link' : '沒有上一筆相同 ids 的預測'}
                  >← 上一次預測</button>
                  <button
                    onClick={() => void navigateToPrediction('next')}
                    disabled={loading || navigationBusy || neighborLoading || !predictionNeighbors.next?.tenhou_link}
                    title={predictionNeighbors.next?.tenhou_link ? `前往 CSV 第 ${predictionNeighbors.next.source_row_number} 列` : predictionNeighbors.next ? '下一筆預測缺少 tenhou_link' : '沒有下一筆相同 ids 的預測'}
                  >下一次預測 →</button>
                  <input
                    value={sideUrl}
                    onChange={e => {
                      neighborAbortRef.current?.abort()
                      neighborAbortRef.current = null
                      navigationAbortRef.current?.abort()
                      navigationAbortRef.current = null
                      neighborRequestRef.current += 1
                      navigationRequestRef.current += 1
                      pendingTableScrollRef.current = null
                      setNeighborLoading(false)
                      setNavigationBusy(false)
                      setSideUrl(e.target.value)
                      setPredictionNeighbors({ previous: null, next: null })
                    }}
                    onKeyDown={e => { if (e.key === 'Enter') setSideUrl(normalizeUrl(sideUrl)) }}
                    style={{ flex: 1, minWidth: '180px' }}
                  />
                  <button onClick={() => sideUrl && openLinkExternal(sideUrl)}>Open External</button>
                </>
              ) : (
                <>
                  <select value={selectedImage?.id ?? ''} onChange={e => setSelectedImageId(e.target.value)} style={{ flex: 1, minWidth: '180px' }}>
                    {activeSession?.images.map(img => <option key={img.id} value={img.id}>{img.name}</option>)}
                  </select>
                  <button onClick={() => setImageZoom(z => Math.max(0.2, z - 0.1))}>-</button>
                  <button onClick={() => setImageZoom(1)}>100%</button>
                  <button onClick={() => setImageZoom(z => Math.min(5, z + 0.1))}>+</button>
                  <span>{Math.round(imageZoom * 100)}%</span>
                </>
              )}
              <button onClick={() => setSidePaneOpen(false)}>Close</button>
            </div>
            <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
              {sideMode === 'web' ? (
                sideUrl ? <iframe key={sideUrl} src={sideUrl} title="CSV link preview" style={{ width: '100%', height: '100%', border: 0 }} /> : <div style={{ padding: '24px', color: '#888' }}>Double click a link cell to preview here.</div>
              ) : (
                selectedImage ? <img src={selectedImage.url} alt={selectedImage.name} style={{ transform: `scale(${imageZoom})`, transformOrigin: 'top left', maxWidth: '100%', margin: '8px' }} /> : <div style={{ padding: '24px', color: '#888' }}>No PNG image found.</div>
              )}
            </div>
          </aside>
          </>
        )}
      </div>

      {activeSession && (
        <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
          <button onClick={copySelectedCell} disabled={!selectedCell}>Copy Cell</button>
          <button onClick={copySelectedRow} disabled={selectedRowIndex === null}>Copy Row As CSV / Ctrl+C</button>
          {!focusMode && activeSession.images.length > 0 && <button onClick={() => { setSidePaneOpen(true); setSideMode('image') }}>Show Training PNGs ({activeSession.images.length})</button>}
        </div>
      )}
    </div>
  )
}
