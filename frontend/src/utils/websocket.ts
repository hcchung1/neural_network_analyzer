import { useState, useEffect, useCallback } from 'react'

export type WSMessage = {
  action: string
  [key: string]: unknown
}

export type WSStatus = 'connecting' | 'open' | 'closed' | 'error'

export function useWebSocket(url: string) {
  const [ws, setWs] = useState<WebSocket | null>(null)
  const [status, setStatus] = useState<WSStatus>('connecting')
  const [lastMessage, setLastMessage] = useState<unknown>(null)

  const connect = useCallback(() => {
    const socket = new WebSocket(url)

    socket.onopen = () => {
      setStatus('open')
    }

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        setLastMessage(data)
      } catch {
        setLastMessage(event.data)
      }
    }

    socket.onclose = () => {
      setStatus('closed')
    }

    socket.onerror = () => {
      setStatus('error')
    }

    setWs(socket)
  }, [url])

  useEffect(() => {
    connect()
    return () => {
      ws?.close()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connect])

  const send = useCallback(
    (msg: WSMessage) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(msg))
      }
    },
    [ws]
  )

  return { status, lastMessage, send, connect }
}
