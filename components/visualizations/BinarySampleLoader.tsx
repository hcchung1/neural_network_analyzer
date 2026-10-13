import { useEffect, useRef, useState } from 'react'

const API_BASE_URL = ''

interface ModelShape { seq_len: number; feature_dim: number }
interface Props {
  modelShape: ModelShape | null
  onFeatureLoaded: (feature: number[][], tokenCount: number, meta?: Record<string, unknown>) => void
  onTenhouUrl: (url: string) => void
}

interface BinFile { name: string; size: number }

export default function BinarySampleLoader({ modelShape, onFeatureLoaded, onTenhouUrl }: Props) {
  const lifetime = useRef(0)
  useEffect(() => {
    lifetime.current++
    return () => { lifetime.current++ }
  }, [])
  const [directory, setDirectory] = useState('cache_100')
  const [files, setFiles] = useState<BinFile[]>([])
  const [fileName, setFileName] = useState('')
  const [lineNumber, setLineNumber] = useState('')
  const [recordIndex, setRecordIndex] = useState('0')
  const [predId, setPredId] = useState('0')
  const [status, setStatus] = useState('')
  const [busy, setBusy] = useState(false)

  const scan = async () => {
    const generation = lifetime.current
    setBusy(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/legacy/binary_samples/files?directory=${encodeURIComponent(directory)}`)
      const data = await response.json()
      if (generation !== lifetime.current) return
      if (!data.success) throw new Error(data.error || 'Scan failed')
      setFiles(data.files)
      setFileName((current) => current && data.files.some((f: BinFile) => f.name === current) ? current : (data.files[0]?.name ?? ''))
      setStatus(`找到 ${data.files.length.toLocaleString()} 個 .bin 檔案`)
    } catch (error) {
      if (generation !== lifetime.current) return
      setStatus(`掃描失敗：${error instanceof Error ? error.message : String(error)}`)
    } finally {
      if (generation === lifetime.current) setBusy(false)
    }
  }

  useEffect(() => { void scan() }, [])

  const load = async () => {
    const generation = lifetime.current
    setBusy(true)
    setStatus('解析樣本中…')
    try {
      const body = {
        directory,
        file_name: fileName,
        line_number: lineNumber.trim() === '' ? null : Number(lineNumber),
        record_index: recordIndex.trim() === '' ? null : Number(recordIndex),
        pred_id: Number(predId),
        seq_len: modelShape?.seq_len ?? null,
        feature_dim: modelShape?.feature_dim ?? null
      }
      const response = await fetch(`${API_BASE_URL}/api/legacy/binary_samples/search`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
      })
      const data = await response.json()
      if (generation !== lifetime.current) return
      if (!data.success) throw new Error(data.error || 'Search failed')
      onTenhouUrl(data.tenhou_url || '')
      if (data.compatible && Array.isArray(data.feature)) {
        onFeatureLoaded(data.feature, data.valid_len, {
          file_name: data.file_name,
          record_index: data.record_index,
          line_number: data.line_number,
          player_id: data.player_id,
          pred_id: body.pred_id,
          target_seat: data.target_seat,
          seat_encoding: data.seat_encoding,
          valid_len: data.valid_len,
          label: data.label,
          conversion: data.conversion,
          shape: data.shape,
        })
        setStatus(`${data.file_name} #${data.record_index} · line ${data.line_number} · label ${data.label} · shape [${data.shape.join(', ')}]`)
      } else {
        setStatus(`已找到 ${data.file_name} #${data.record_index}，但 ${data.warning}`)
      }
    } catch (error) {
      if (generation !== lifetime.current) return
      setStatus(`載入失敗：${error instanceof Error ? error.message : String(error)}`)
    } finally {
      if (generation === lifetime.current) setBusy(false)
    }
  }

  return <section className="mb-5 rounded-xl border border-slate-200 bg-slate-50 p-4">
    <strong className="mb-3 block text-sm text-slate-800">Binary Sample Search</strong>
    <div className="grid gap-2 [&_input]:min-w-0 [&_input]:rounded-lg [&_input]:border [&_input]:border-slate-300 [&_input]:bg-white [&_input]:px-3 [&_input]:py-2 [&_input]:text-sm [&_input]:focus:outline-none [&_input]:focus:ring-2 [&_input]:focus:ring-blue-500 [&_select]:min-w-0 [&_select]:rounded-lg [&_select]:border [&_select]:border-slate-300 [&_select]:bg-white [&_select]:px-3 [&_select]:py-2 [&_select]:text-sm [&_select]:focus:outline-none [&_select]:focus:ring-2 [&_select]:focus:ring-blue-500 [&_button]:rounded-lg [&_button]:border [&_button]:border-slate-300 [&_button]:bg-white [&_button]:px-3 [&_button]:py-2 [&_button]:text-sm [&_button]:font-medium [&_button]:text-slate-700 [&_button]:hover:bg-slate-100 [&_button]:disabled:opacity-50">
      <div className="flex flex-wrap gap-2">
        <input aria-label="Binary directory" value={directory} onChange={e => setDirectory(e.target.value)} className="flex-1" />
        <button onClick={scan} disabled={busy}>Scan</button>
      </div>
      <select aria-label="Binary file" value={fileName} onChange={e => setFileName(e.target.value)}>
        <option value="">All .bin files (line search)</option>
        {files.map(file => <option key={file.name} value={file.name}>{file.name} ({(file.size / 1024).toFixed(1)} KiB)</option>)}
      </select>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(120px,auto)]">
        <input aria-label="Line number" type="number" min="0" placeholder="line (optional)" value={lineNumber} onChange={e => setLineNumber(e.target.value)} />
        <input aria-label="Record index" type="number" min="0" placeholder="record" value={recordIndex} onChange={e => setRecordIndex(e.target.value)} />
        <select aria-label="Prediction opponent" value={predId} onChange={e => setPredId(e.target.value)}>
          <option value="0">next opponent</option><option value="1">opposite opponent</option><option value="2">previous opponent</option>
        </select>
      </div>
      <button className="!border-blue-600 !bg-blue-600 !text-white hover:!bg-blue-700" onClick={load} disabled={busy || (!fileName && !lineNumber.trim())}>{busy ? 'Working…' : 'Load Sample'}</button>
      {status && <small style={{ color: status.includes('失敗') || status.includes('但') ? '#c92a2a' : '#495057', overflowWrap: 'anywhere' }}>{status}</small>}
    </div>
  </section>
}
