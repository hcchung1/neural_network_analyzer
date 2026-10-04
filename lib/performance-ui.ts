export const CSV_ROW_HEIGHT = 30
export const CSV_HEADER_HEIGHT = 32
export const OUTPUT_PREVIEW_VALUES = 100

export function rowWindow(count: number, scrollTop: number, height: number, rowHeight = CSV_ROW_HEIGHT, overscan = 8) {
  const first = Math.max(0, Math.floor(scrollTop / rowHeight))
  const start = Math.min(count, Math.max(0, first - overscan))
  const end = Math.min(count, first + Math.ceil(height / rowHeight) + overscan)
  return { start, end, before: start * rowHeight, after: (count - end) * rowHeight }
}

export function attentionMatrix(attention: number[][] | number[][][] | null) {
  const source = attention || []
  // Empty heads may precede nonempty heads, so inspect the first actual row.
  const multiHead = source.some(row => row.length > 0 && Array.isArray(row[0]))
  const heads: number[][][] = multiHead ? source as number[][][] : [source as number[][]]
  let rows = 0
  let columns = 0
  for (const head of heads) {
    rows = Math.max(rows, head.length)
    for (const row of head) columns = Math.max(columns, row.length)
  }
  let min = Infinity
  let max = -Infinity
  let allZero = true
  const values: number[][] = []
  for (let row = 0; row < rows; row++) {
    const cells: number[] = []
    for (let column = 0; column < columns; column++) {
      let sum = 0
      for (const head of heads) sum += head[row]?.[column] || 0
      const value = sum / heads.length
      cells.push(value)
      min = Math.min(min, value)
      max = Math.max(max, value)
      if (Math.abs(value) >= 1e-10) allZero = false
    }
    values.push(cells)
  }
  return { values, rows, columns, min: min === Infinity ? 0 : min, max: max === -Infinity ? 0 : max, allZero }
}

export function attentionCell(x: number, y: number, width: number, height: number, rows: number, columns: number) {
  if (x < 0 || y < 0 || x >= width || y >= height || !rows || !columns) return null
  return { row: Math.floor(y / height * rows), column: Math.floor(x / width * columns) }
}

export function outputPreview(values: number[]) {
  return values.slice(0, OUTPUT_PREVIEW_VALUES)
}
