export async function copyImageToClipboard(imageUrl: string): Promise<void> {
  if (!navigator.clipboard?.write || typeof ClipboardItem === 'undefined') {
    throw new Error('此瀏覽器或連線不支援圖片複製，請使用 HTTPS 或 localhost 開啟頁面。')
  }
  const image = fetch(imageUrl, { cache: 'no-store' }).then(async response => {
    if (!response.ok) {
      const details = await response.json().catch(() => null)
      throw new Error(details?.error || '無法讀取圖片')
    }
    const blob = await response.blob()
    if (blob.type !== 'image/png') throw new Error('圖片不是 PNG 格式')
    return blob
  })
  // Start the clipboard write during the click gesture, before the fetch resolves.
  // Safari requires this even though the ClipboardItem may contain a promise.
  void image.catch(() => {})
  try {
    await navigator.clipboard.write([new ClipboardItem({ 'image/png': image })])
  } catch (error) {
    // Retain the file error when writing fails because its image promise rejected.
    await image
    throw new Error(error instanceof DOMException && error.name === 'NotAllowedError'
      ? '瀏覽器拒絕圖片複製，請允許剪貼簿權限後再試一次。'
      : '圖片複製失敗，請再試一次。')
  }
}

export async function copyToClipboard(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text)
      return
    } catch {
      // Some browsers expose the API but deny clipboard permission.
    }
  }

  // Remote HTTP pages may not expose the Clipboard API at all.
  const activeElement = document.activeElement
  const selection = window.getSelection()
  const ranges = selection
    ? Array.from({ length: selection.rangeCount }, (_, index) => selection.getRangeAt(index).cloneRange())
    : []
  const textarea = document.createElement('textarea')
  textarea.value = text
  textarea.readOnly = true
  textarea.style.position = 'fixed'
  textarea.style.top = '0'
  textarea.style.left = '-9999px'
  document.body.appendChild(textarea)

  try {
    textarea.focus({ preventScroll: true })
    textarea.select()
    if (!document.execCommand('copy')) {
      throw new Error('瀏覽器不允許複製，請使用 HTTPS 或 localhost 開啟頁面後再試。')
    }
  } finally {
    textarea.remove()
    if (activeElement instanceof HTMLElement) activeElement.focus({ preventScroll: true })
    if (selection) {
      selection.removeAllRanges()
      ranges.forEach(range => selection.addRange(range))
    }
  }
}
