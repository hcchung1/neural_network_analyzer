export type HealthState = { status: 'checking' | 'healthy' | 'offline'; rttMs: number | null }

interface HealthMonitorOptions {
  fetch: (url: string, options: RequestInit) => Promise<{ ok: boolean }>
  now: () => number
  onChange: (state: HealthState) => void
  isVisible: () => boolean
  intervalMs?: number
  timeoutMs?: number
}

// The hook owns DOM events; this controller owns scheduling and request lifetime.
export function createHealthMonitor(options: HealthMonitorOptions) {
  let disposed = false
  let active: AbortController | null = null
  let timer: ReturnType<typeof setTimeout> | undefined
  let recheck = false
  const clearTimer = () => { if (timer !== undefined) clearTimeout(timer); timer = undefined }
  const schedule = () => {
    clearTimer()
    if (!disposed && options.isVisible()) timer = setTimeout(() => void check(), options.intervalMs ?? 30000)
  }
  async function check() {
    if (disposed || !options.isVisible() || active) return
    clearTimer()
    const controller = new AbortController()
    active = controller
    const started = options.now()
    let timedOut = false
    const deadline = setTimeout(() => { timedOut = true; controller.abort() }, options.timeoutMs ?? 10000)
    try {
      const response = await options.fetch('/api/v1/health', { signal: controller.signal, cache: 'no-store' })
      if (!disposed && !controller.signal.aborted) {
        options.onChange(response.ok
          ? { status: 'healthy', rttMs: Math.max(0, Math.round(options.now() - started)) }
          : { status: 'offline', rttMs: null })
      }
    } catch {
      if (!disposed && (!controller.signal.aborted || timedOut)) options.onChange({ status: 'offline', rttMs: null })
    } finally {
      clearTimeout(deadline)
      active = null
      if (recheck && !disposed && options.isVisible()) { recheck = false; void check() }
      else schedule()
    }
  }
  function visibilityChanged() {
    clearTimer()
    if (!options.isVisible()) { active?.abort(); return }
    if (active) { recheck = true; return }
    return check()
  }
  return { check, visibilityChanged, dispose() { disposed = true; clearTimer(); active?.abort() } }
}
