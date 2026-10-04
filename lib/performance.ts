/** Ordered results with bounded outstanding operations, suitable for file reads. */
export async function mapWithConcurrency<T, R>(
  values: T[], concurrency: number, operation: (value: T, index: number) => Promise<R>,
): Promise<R[]> {
  if (!Number.isInteger(concurrency) || concurrency <= 0) throw new Error('Concurrency must be a positive integer');
  const results = new Array<R>(values.length);
  let cursor = 0;
  await Promise.all(Array.from({ length: Math.min(concurrency, values.length) }, async () => {
    while (cursor < values.length) {
      const index = cursor++;
      results[index] = await operation(values[index], index);
    }
  }));
  return results;
}

/** Stop sizing once the metadata budget is exceeded; never stringify large data. */
export function exceedsStorageBudget(value: unknown, budget = 64 * 1024): boolean {
  const pending: unknown[] = [value];
  const visited = new WeakSet<object>();
  let bytes = 0;
  while (pending.length && bytes <= budget) {
    const item = pending.pop();
    if (typeof item === 'string') bytes += item.length * 2;
    else if (typeof item === 'number') bytes += 8;
    else if (item && typeof item === 'object' && !visited.has(item)) {
      visited.add(item);
      // Avoid copying a potentially huge array to the pending stack.
      if (Array.isArray(item)) {
        if (item.length * 8 > budget - bytes) return true;
        for (const entry of item) pending.push(entry);
      } else {
        for (const [key, entry] of Object.entries(item)) { bytes += key.length * 2; pending.push(entry); }
      }
    } else bytes += 4;
  }
  return bytes > budget;
}
