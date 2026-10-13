import { NextRequest, NextResponse } from 'next/server'
import { readFile, readdir, realpath, stat } from 'node:fs/promises'
import path from 'node:path'
import { homedir } from 'node:os'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

export async function GET(req: NextRequest) {
  const folder = req.nextUrl.searchParams.get('folder')
  const run = req.nextUrl.searchParams.get('run')
  const kind = req.nextUrl.searchParams.get('kind')
  const checking = req.nextUrl.searchParams.get('check') === '1'
  const presence = (exists: boolean) => NextResponse.json({ exists }, { headers: { 'Cache-Control': 'no-store' } })
  if (!folder || !run || /[/\\\x00]/.test(run) || run === '.' || run === '..' || !['training', 'phase'].includes(kind || '')) {
    return NextResponse.json({ error: '圖片資料夾、Run 名稱或圖片類型無效' }, { status: 400 })
  }

  try {
    const expandedFolder = folder.replace(/^~(?=$|[/\\])/, homedir())
    const root = await realpath(path.resolve(process.cwd(), 'backend', expandedFolder))
    if (!(await stat(root)).isDirectory()) {
      if (checking) return presence(false)
      return NextResponse.json({ error: '圖片資料夾不存在' }, { status: 404 })
    }
    // Imported CSV names already include `_all`; bare Run tags are also accepted.
    const stem = run.endsWith('_all') ? run : `${run}_all`
    const filename = kind === 'training' ? `${stem}.png` : `${stem}_base_vs_random_turn_metrics.png`
    const relativeFile = kind === 'training' ? filename : path.join('random_feature_analysis', filename)
    const runFolder = run.replace(/_all$/, '')
    let imagePath: string | undefined
    for (const candidate of [path.join(root, relativeFile), path.join(root, runFolder, relativeFile)]) {
      try {
        const resolved = await realpath(candidate)
        if (resolved.startsWith(root + path.sep) && (await stat(resolved)).isFile()) {
          imagePath = resolved
          break
        }
      } catch (error) {
        if (!['ENOENT', 'ENOTDIR'].includes((error as NodeJS.ErrnoException).code || '')) throw error
      }
    }
    // Folder scans are recursive, so also support runs inside additional subfolders.
    const directories = imagePath ? [] : [root]
    while (directories.length && !imagePath) {
      const directory = directories.pop()!
      for (const entry of await readdir(directory, { withFileTypes: true })) {
        if (entry.isDirectory()) directories.push(path.join(directory, entry.name))
        if (entry.isFile() && entry.name === filename && (kind === 'training' || path.basename(directory) === 'random_feature_analysis')) {
          imagePath = path.join(directory, entry.name)
          break
        }
      }
    }
    if (checking) return presence(Boolean(imagePath))
    if (!imagePath) return NextResponse.json({ error: `找不到圖片：${relativeFile}` }, { status: 404 })
    const bytes = await readFile(imagePath)
    if (!bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) {
      return NextResponse.json({ error: '圖片不是有效的 PNG 檔案' }, { status: 422 })
    }
    return new NextResponse(new Uint8Array(bytes), {
      headers: { 'Content-Type': 'image/png', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' },
    })
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code
    const missing = code === 'ENOENT' || code === 'ENOTDIR'
    if (checking && missing) return presence(false)
    return NextResponse.json({ error: missing ? '找不到圖片資料夾或圖片檔案' : '無法讀取圖片檔案' }, { status: missing ? 404 : 500 })
  }
}
