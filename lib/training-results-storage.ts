import { exceedsStorageBudget } from './performance';

const DATABASE = 'archanalyzer-training-results';
const STORE = 'results';
let writes: Promise<void> = Promise.resolve();

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') { reject(new Error('IndexedDB is unavailable')); return; }
    const request = indexedDB.open(DATABASE, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE);
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error('Training storage is blocked by another tab'));
    request.onsuccess = () => {
      request.result.onversionchange = () => request.result.close();
      resolve(request.result);
    };
  });
}

export async function readTrainingResults<T>(key: string): Promise<T[]> {
  if (typeof window === 'undefined') return [];
  await writes.catch(() => undefined);
  const stored = localStorage.getItem(key);
  if (!stored) return [];
  const parsed = JSON.parse(stored);
  if (Array.isArray(parsed)) return parsed; // Existing saved results remain readable.
  if (parsed?.version !== 2 || parsed?.storage !== 'indexeddb') return [];
  const db = await openDatabase();
  try {
    return await new Promise<T[]>((resolve, reject) => {
      const transaction = db.transaction(STORE, 'readonly');
      const request = transaction.objectStore(STORE).get(key);
      transaction.oncomplete = () => Array.isArray(request.result)
        ? resolve(request.result) : reject(new Error('Saved training results are unavailable'));
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error || new Error('Training storage read aborted'));
    });
  } finally { db.close(); }
}

export function writeTrainingResults<T>(key: string, results: T[]): Promise<void> {
  const save = async () => {
    if (typeof window === 'undefined') return;
    if (!exceedsStorageBudget(results)) {
      localStorage.setItem(key, JSON.stringify(results));
      return;
    }
    const db = await openDatabase();
    try {
      await new Promise<void>((resolve, reject) => {
        const transaction = db.transaction(STORE, 'readwrite');
        transaction.objectStore(STORE).put(results, key);
        transaction.oncomplete = () => resolve();
        transaction.onerror = () => reject(transaction.error);
        transaction.onabort = () => reject(transaction.error || new Error('Training storage write aborted'));
      });
      // Publish the small pointer only after the durable transaction succeeds.
      localStorage.setItem(key, JSON.stringify({ version: 2, storage: 'indexeddb' }));
    } finally { db.close(); }
  };
  const pending = writes.catch(() => undefined).then(save);
  writes = pending;
  return pending;
}
