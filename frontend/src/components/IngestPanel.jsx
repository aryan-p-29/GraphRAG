import { useState, useRef, useEffect } from 'react'
import { Play, Plus, X, FolderInput } from 'lucide-react'

const DEFAULT_EXTS = ['.py']

export default function IngestPanel({ onDone }) {
  const [source, setSource]   = useState('')
  const [exts, setExts]       = useState(DEFAULT_EXTS)
  const [extInput, setExtInput] = useState('')
  const [status, setStatus]   = useState('idle')   // idle | running | done | error
  const [logs, setLogs]       = useState([])
  const logRef = useRef(null)

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logs])

  function addExt() {
    const e = extInput.trim()
    if (!e) return
    const norm = e.startsWith('.') ? e : `.${e}`
    if (!exts.includes(norm)) setExts(prev => [...prev, norm])
    setExtInput('')
  }

  async function handleIngest() {
    if (!source.trim() || status === 'running') return
    setLogs([])
    setStatus('running')

    try {
      const res = await fetch('/api/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: source.trim(), extensions: exts }),
      })

      if (!res.ok) {
        const err = await res.json()
        setLogs(l => [...l, `❌ ${err.detail}`])
        setStatus('error')
        return
      }

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      // Process one SSE chunk string → returns true if a terminal __STATUS__ was seen
      function processChunk(chunk) {
        const line = chunk.replace(/^data:\s*/, '').trim()
        if (!line) return false
        if (line.startsWith('__STATUS__')) {
          const s = line.replace('__STATUS__', '')
          setStatus(s)
          if (s === 'done') onDone?.()
          return true   // terminal — stop reading
        }
        setLogs(l => [...l, line])
        return false
      }

      let finished = false
      while (!finished) {
        const { value, done } = await reader.read()

        // Append newly received bytes to the buffer
        if (value) buf += decoder.decode(value, { stream: !done })

        // Split on SSE event boundaries (\n\n)
        const parts = buf.split('\n\n')
        // Keep the last (possibly incomplete) segment in the buffer
        buf = parts.pop() ?? ''

        for (const chunk of parts) {
          if (processChunk(chunk)) { finished = true; break }
        }

        // Stream closed — flush whatever is still in buf
        if (done && !finished) {
          if (buf.trim()) processChunk(buf)
          buf = ''
          finished = true
        }
      }

      // Safety net: if the stream closed without an explicit __STATUS__ event,
      // poll the status endpoint once to get the definitive answer.
      setStatus(prev => {
        if (prev === 'running') {
          fetch('/api/ingest/status')
            .then(r => r.json())
            .then(d => {
              setStatus(d.status)
              if (d.status === 'done') onDone?.()
            })
            .catch(() => setStatus('error'))
          return prev  // will be replaced by the fetch above
        }
        return prev
      })

    } catch (err) {
      setLogs(l => [...l, `❌ Network error: ${err.message}`])
      setStatus('error')
    }
  }

  function classForLine(line) {
    if (line.startsWith('✅')) return 'log-line done'
    if (line.startsWith('❌')) return 'log-line error'
    if (line.startsWith('✔'))  return 'log-line ok'
    return 'log-line'
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* Source input */}
      <div className="field">
        <label>Repository source</label>
        <div style={{ position: 'relative' }}>
          <FolderInput
            size={13}
            style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }}
          />
          <input
            className="input"
            style={{ paddingLeft: 30 }}
            placeholder="GitHub URL or /local/path"
            value={source}
            onChange={e => setSource(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleIngest()}
          />
        </div>
      </div>

      {/* Extensions */}
      <div className="field">
        <label>File extensions</label>
        <div className="tag-list" style={{ marginBottom: 6 }}>
          {exts.map(ext => (
            <span key={ext} className="tag">
              {ext}
              <button onClick={() => setExts(prev => prev.filter(e => e !== ext))}>
                <X size={10} />
              </button>
            </span>
          ))}
        </div>
        <div className="tag-input">
          <input
            className="input"
            style={{ flex: 1 }}
            placeholder=".js .go …"
            value={extInput}
            onChange={e => setExtInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); addExt() } }}
          />
          <button className="btn btn-ghost" onClick={addExt} title="Add extension">
            <Plus size={14} />
          </button>
        </div>
      </div>

      {/* Status + ingest button */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
        <span className={`status-badge ${status}`}>
          <span className="status-dot" />
          {status === 'idle'    && 'Idle'}
          {status === 'running' && 'Ingesting…'}
          {status === 'done'    && 'Complete'}
          {status === 'error'   && 'Error'}
        </span>
        <button
          className="btn btn-primary"
          disabled={!source.trim() || status === 'running'}
          onClick={handleIngest}
        >
          <Play size={13} />
          {status === 'running' ? 'Running…' : 'Ingest'}
        </button>
      </div>

      {/* Log stream */}
      {logs.length > 0 && (
        <div className="log-stream" ref={logRef}>
          {logs.map((line, i) => (
            <div key={i} className={classForLine(line)}>{line}</div>
          ))}
        </div>
      )}
    </div>
  )
}
