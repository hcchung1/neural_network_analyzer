import { useState, useEffect, useCallback, useRef } from 'react'

export type WSMessage = {
  action: string
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any
}

export type WSStatus = 'connecting' | 'open' | 'closed' | 'error'

const MAX_RECONNECT_ATTEMPTS = 10
const BASE_RECONNECT_DELAY = 1000 // 1 second

export function useWebSocket(url: string, onMessage?: (data: WSMessage) => void, enabled = true) {
  const wsRef = useRef<WebSocket | null>(null)
  const [status, setStatus] = useState<WSStatus>(enabled ? 'connecting' : 'closed')
  const [lastMessage, setLastMessage] = useState<unknown>(null)
  const reconnectAttempts = useRef(0)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isManualClose = useRef(false)
  const onMessageRef = useRef(onMessage)

  // Update ref when onMessage changes
  useEffect(() => {
    onMessageRef.current = onMessage
  }, [onMessage])

  const connect = useCallback(() => {
    if (!enabled) {
      setStatus('closed')
      return
    }
    if (isManualClose.current) return
    if (reconnectAttempts.current >= MAX_RECONNECT_ATTEMPTS) {
      console.error(`[WebSocket] Max reconnect attempts (${MAX_RECONNECT_ATTEMPTS}) reached`)
      setStatus('error')
      return
    }

    try {
      setStatus('connecting')
      const socket = new WebSocket(url)

      socket.onopen = () => {
        console.log(`[WebSocket] Connected to ${url}`)
        setStatus('open')
        reconnectAttempts.current = 0 // Reset on successful connection
      }

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          setLastMessage(data)
          // Also call the onMessage callback if provided
          if (onMessageRef.current) {
            onMessageRef.current(data)
          }
        } catch {
          setLastMessage(event.data)
        }
      }

      socket.onclose = (event) => {
        console.log(`[WebSocket] Connection closed (code: ${event.code}, reason: ${event.reason || 'N/A'})`)
        setStatus('closed')
        
        // Auto-reconnect with exponential backoff
        if (!isManualClose.current) {
          const delay = Math.min(
            BASE_RECONNECT_DELAY * Math.pow(2, reconnectAttempts.current),
            30000 // Max 30 seconds
          )
          reconnectAttempts.current += 1
          console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current}/${MAX_RECONNECT_ATTEMPTS})...`)
          
          reconnectTimer.current = setTimeout(() => {
            connect()
          }, delay)
        }
      }

      socket.onerror = (error) => {
        console.error('[WebSocket] Error:', error)
        setStatus('error')
      }

      wsRef.current = socket
    } catch (err) {
      console.error('[WebSocket] Failed to create connection:', err)
      setStatus('error')
    }
  }, [enabled, url])

  useEffect(() => {
    if (!enabled) {
      isManualClose.current = true
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current)
        reconnectTimer.current = null
      }
      wsRef.current?.close()
      wsRef.current = null
      setStatus('closed')
      return
    }

    isManualClose.current = false
    connect()
    
    return () => {
      isManualClose.current = true
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current)
        reconnectTimer.current = null
      }
      wsRef.current?.close()
      wsRef.current = null
    }
  }, [connect, enabled])

  const send = useCallback(
    (msg: WSMessage) => {
      const socket = wsRef.current
      if (socket && socket.readyState === WebSocket.OPEN) {
        console.log('[WebSocket] Sending:', msg.action, 'bufferedAmount:', socket.bufferedAmount)
        try {
          const data = JSON.stringify(msg)
          socket.send(data)
          console.log('[WebSocket] Sent successfully:', msg.action, 'data length:', data.length)
        } catch (err) {
          console.error('[WebSocket] Failed to send message:', msg.action, err)
        }
      } else {
        console.warn('[WebSocket] Cannot send message, connection not open. Status:', status, 'readyState:', socket?.readyState)
      }
    },
    [status]
  )

  const reconnect = useCallback(() => {
    reconnectAttempts.current = 0
    isManualClose.current = false
    connect()
  }, [connect])

  return { status, lastMessage, send, reconnect }
}
