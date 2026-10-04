'use client'
import { useEffect, useState } from 'react'
import { createHealthMonitor, type HealthState } from '@/lib/health-monitor'

export function useSidecarHealth(): HealthState {
  const [health, setHealth] = useState<HealthState>({ status: 'checking', rttMs: null })
  useEffect(() => {
    const monitor = createHealthMonitor({
      fetch: (url, options) => fetch(url, options), now: () => performance.now(),
      onChange: setHealth, isVisible: () => document.visibilityState !== 'hidden',
    })
    const onVisibility = () => { void monitor.visibilityChanged() }
    document.addEventListener('visibilitychange', onVisibility)
    void monitor.check()
    return () => { document.removeEventListener('visibilitychange', onVisibility); monitor.dispose() }
  }, [])
  return health
}
