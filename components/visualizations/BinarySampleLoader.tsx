import { useEffect, useState } from 'react'

const API_BASE_URL = ''

interface ModelShape { seq_len: number; feature_dim: number }
interface Props {
  modelShape: ModelShape | null
  onFeatureLoaded: (feature: number[][], tokenCount: number, meta?: Record<string, unknown>) => void
  onTenhouUrl: (url: string) => void
}

interface BinFile { name: string; size: number }

export default function BinarySampleLoader({ modelShape, onFeatureLoaded, onTenhouUrl }: Props) {
  const [directory, setDirectory] = useState('cache_100')
  const [files, setFiles] = useState<BinFile[]>([])
  const [fileName, setFileName] = useState('')
  const [lineNumber, setLineNumber] = useState('')
  const [recordIndex, setRecordIndex] = useState('0')
  const [predId, setPredId] = useState('0')
  const [status, setStatus] = useState('')
  const [busy, setBusy] = useState(false)

  const scan = async () => {
    setBusy(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/legacy/binary_samples/files?directory=${encodeURIComponent(directory)}`)
      const data = await response.json()
      if (!data.success) throw new Error(data.error || 'Scan failed')
      setFiles(data.files)
      setFileName((current) => current && data.files.some((f: BinFile) => f.name === current) ? current : (data.files[0]?.name ?? ''))
      setStatus(`找到 ${data.files.length.toLocaleString()} 個 .bin 檔案`)
    } catch (error) {
      setStatus(`掃描失敗：${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => { void scan() }, [])

  const load = async () => {
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
      setStatus(`載入失敗：${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setBusy(false)
    }
  }

  return <section style={{ marginBottom: 20, padding: 12, border: '1px solid #ced4da', borderRadius: 6, background: '#fff' }}>
    <strong style={{ display: 'block', marginBottom: 8 }}>Binary Sample Search</strong>
    <div style={{ display: 'grid', gap: 7 }}>
      <div style={{ display: 'flex', gap: 6 }}>
        <input aria-label="Binary directory" value={directory} onChange={e => setDirectory(e.target.value)} style={{ flex: 1, minWidth: 0 }} />
        <button onClick={scan} disabled={busy}>Scan</button>
      </div>
      <select aria-label="Binary file" value={fileName} onChange={e => setFileName(e.target.value)}>
        <option value="">All .bin files (line search)</option>
        {files.map(file => <option key={file.name} value={file.name}>{file.name} ({(file.size / 1024).toFixed(1)} KiB)</option>)}
      </select>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 72px', gap: 6 }}>
        <input aria-label="Line number" type="number" min="0" placeholder="line (optional)" value={lineNumber} onChange={e => setLineNumber(e.target.value)} />
        <input aria-label="Record index" type="number" min="0" placeholder="record" value={recordIndex} onChange={e => setRecordIndex(e.target.value)} />
        <select aria-label="Prediction opponent" value={predId} onChange={e => setPredId(e.target.value)}>
          <option value="0">next opponent</option><option value="1">opposite opponent</option><option value="2">previous opponent</option>
        </select>
      </div>
      <button onClick={load} disabled={busy || (!fileName && !lineNumber.trim())}>{busy ? 'Working…' : 'Load Sample'}</button>
      {status && <small style={{ color: status.includes('失敗') || status.includes('但') ? '#c92a2a' : '#495057', overflowWrap: 'anywhere' }}>{status}</small>}
    </div>
  </section>
}
