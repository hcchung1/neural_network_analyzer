import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';
import crypto from 'crypto';

const DATA_DIR = path.join(process.cwd(), 'data');
const DB_PATH = path.join(DATA_DIR, 'catalog.sqlite3');
const LEGACY_OVERRIDES_PATH = path.join(DATA_DIR, 'legacy-overrides.json');

function ensureDataDir() {
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }
}

function classifyArtifactKind(fileName: string): string {
  const n = fileName.toLowerCase();
  if (n.endsWith("_train_state.pth") || n.includes("train_state")) return "train_state";
  if (n.endsWith(".pth") || n.endsWith(".pt")) {
    if (n.includes("best")) return "best_checkpoint";
    if (n.includes("epoch")) return "epoch_checkpoint";
    return "checkpoint";
  }
  if (n.endsWith("_history.csv")) return "training_history";
  if (n.includes("false_positive") || n.endsWith("_fp.csv")) return "false_positive";
  if (n.includes("false_negative") || n.endsWith("_fn.csv")) return "false_negative";
  if (n.includes("random_feature") && (n.includes("pred") || n.includes("prediction"))) return "random_feature_prediction";
  if (n.includes("random_feature")) return "random_feature_analysis";
  if (n.includes("phase_summary")) return "phase_summary";
  if (n.includes("turn_metrics")) return "turn_metrics";
  if (n.includes("label_turn")) return "label_turn_statistics";
  if (n.includes("comparison_table")) return "comparison_table";
  if (n.endsWith(".png") || n.endsWith(".jpg") || n.endsWith(".svg")) return "plot";
  if (n.endsWith(".log")) return "log";
  if (["main.py", "transformer.py", "utils.py"].includes(n)) return "code_snapshot";
  if (n.endsWith(".zip") || n.endsWith(".tar") || n.endsWith(".gz") || n.endsWith(".tgz")) return "archive";
  if (n.includes("pred") && n.endsWith(".csv")) return "prediction";
  return "unclassified";
}

let dbInstance: Database.Database | null = null;

export function getDb(): Database.Database {
  if (dbInstance) return dbInstance;
  ensureDataDir();
  
  const db = new Database(DB_PATH);
  
  db.exec(`
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        model_family TEXT,
        created_at REAL,
        manifest_path TEXT,
        is_legacy INTEGER DEFAULT 0,
        confidence TEXT DEFAULT 'verified'
    );

    CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id TEXT PRIMARY KEY,
        run_id TEXT,
        kind TEXT NOT NULL,
        rel_path TEXT NOT NULL,
        abs_path TEXT NOT NULL UNIQUE,
        mtime REAL NOT NULL,
        size INTEGER NOT NULL,
        fingerprint TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS scan_cursors (
        abs_path TEXT PRIMARY KEY,
        mtime REAL NOT NULL,
        size INTEGER NOT NULL,
        fingerprint TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS legacy_overrides (
        run_id TEXT PRIMARY KEY,
        overrides_json TEXT NOT NULL
    );
  `);
  
  dbInstance = db;
  return db;
}

export function loadLegacyOverrides(): Record<string, any> {
  if (!fs.existsSync(LEGACY_OVERRIDES_PATH)) return {};
  try {
    return JSON.parse(fs.readFileSync(LEGACY_OVERRIDES_PATH, 'utf-8'));
  } catch {
    return {};
  }
}

export function saveLegacyOverride(runId: string, overrides: Record<string, any>) {
  ensureDataDir();
  const data = loadLegacyOverrides();
  data[runId] = overrides;
  fs.writeFileSync(LEGACY_OVERRIDES_PATH, JSON.stringify(data, null, 2), 'utf-8');

  const db = getDb();
  const stmt = db.prepare('INSERT OR REPLACE INTO legacy_overrides (run_id, overrides_json) VALUES (?, ?)');
  stmt.run(runId, JSON.stringify(overrides));
}

function computeFingerprint(absPath: string, size: number, mtime: number): string {
  const hash = crypto.createHash('sha256');
  hash.update(`${absPath}:${size}:${mtime}`);
  return hash.digest('hex').substring(0, 16);
}

export async function* scanRootGenerator(rootDir: string) {
  const absRoot = path.resolve(rootDir.startsWith('~') ? rootDir.replace('~', process.env.HOME || '') : rootDir);
  if (!fs.existsSync(absRoot) || !fs.statSync(absRoot).isDirectory()) {
    yield { type: 'error', message: `Root directory not found: ${rootDir}` };
    return;
  }

  const db = getDb();
  const legacyOverrides = loadLegacyOverrides();
  
  const insertOverride = db.prepare('INSERT OR REPLACE INTO legacy_overrides (run_id, overrides_json) VALUES (?, ?)');
  for (const [rid, ov] of Object.entries(legacyOverrides)) {
    insertOverride.run(rid, JSON.stringify(ov));
  }

  const getCursor = db.prepare('SELECT fingerprint FROM scan_cursors WHERE abs_path = ?');
  const insertRun = db.prepare('INSERT OR IGNORE INTO runs (run_id, model_family, created_at, manifest_path) VALUES (?, ?, ?, ?)');
  const insertArtifact = db.prepare(`
    INSERT OR REPLACE INTO artifacts 
    (artifact_id, run_id, kind, rel_path, abs_path, mtime, size, fingerprint) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const insertCursor = db.prepare('INSERT OR REPLACE INTO scan_cursors (abs_path, mtime, size, fingerprint) VALUES (?, ?, ?, ?)');

  let scannedCount = 0;
  const runsCreated = new Set<string>();

  function walkDir(dir: string, callback: (filePath: string) => void) {
    const files = fs.readdirSync(dir);
    for (const file of files) {
      if (file.startsWith('.')) continue; // skip hidden
      const fullPath = path.join(dir, file);
      const stat = fs.statSync(fullPath);
      if (stat.isDirectory()) {
        walkDir(fullPath, callback);
      } else {
        callback(fullPath);
      }
    }
  }

  let totalFiles = 0;
  walkDir(absRoot, () => totalFiles++);

  let currentFile = 0;
  let batchExec = db.transaction(() => {
     // Transaction handled below dynamically
  });

  const filesToProcess: string[] = [];
  walkDir(absRoot, (fullPath) => filesToProcess.push(fullPath));

  for (const absP of filesToProcess) {
    currentFile++;
    if (currentFile % 50 === 0) {
      yield { type: 'progress', progress: currentFile / totalFiles, message: `Scanning ${path.basename(absP)}...` };
    }

    let stat;
    try {
      stat = fs.statSync(absP);
    } catch {
      continue;
    }
    
    const mtime = stat.mtimeMs;
    const size = stat.size;
    const fp = computeFingerprint(absP, size, mtime);

    const row = getCursor.get(absP) as { fingerprint: string } | undefined;
    if (row && row.fingerprint === fp) continue; // Unchanged

    const relP = path.relative(absRoot, absP);
    const kind = classifyArtifactKind(path.basename(absP));

    const parentDir = path.dirname(absP);
    const runId = parentDir === absRoot ? 'root_run' : path.basename(parentDir);

    db.transaction(() => {
      insertRun.run(runId, 'unknown', mtime, null);
      if (!runsCreated.has(runId)) runsCreated.add(runId);

      const artifactId = 'art_' + crypto.createHash('md5').update(absP).digest('hex').substring(0, 12);
      insertArtifact.run(artifactId, runId, kind, relP, absP, mtime, size, fp);
      insertCursor.run(absP, mtime, size, fp);
    })();
    scannedCount++;
  }

  yield { type: 'done', scanned_files: scannedCount, runs_updated: runsCreated.size };
}

export function listRuns(): any[] {
  const db = getDb();
  return db.prepare('SELECT * FROM runs ORDER BY created_at DESC').all();
}

export function getRun(runId: string): any {
  const db = getDb();
  const runRow = db.prepare('SELECT * FROM runs WHERE run_id = ?').get(runId);
  if (!runRow) return null;
  
  const artifacts = db.prepare('SELECT * FROM artifacts WHERE run_id = ?').all(runId);
  return { ...runRow, artifacts };
}
