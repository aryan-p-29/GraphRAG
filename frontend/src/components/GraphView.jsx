import { useEffect, useRef, useState, useCallback } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { Network } from 'lucide-react'

// Node colour by label
const LABEL_COLORS = {
  Class:     '#7c3aed',
  Function:  '#2563eb',
  Method:    '#0891b2',
  Module:    '#d97706',
  Interface: '#16a34a',
  Struct:    '#dc2626',
  TypeAlias: '#9333ea',
  File:      '#6b7280',
}

const EDGE_COLORS = {
  CALLS:     'rgba(124,58,237,0.55)',
  IMPORTS:   'rgba(88,166,255,0.45)',
  DEFINES:   'rgba(63,185,80,0.45)',
  HAS_METHOD:'rgba(56,189,248,0.4)',
  INHERITS:  'rgba(249,115,22,0.5)',
  IMPLEMENTS:'rgba(234,179,8,0.4)',
}

const LEGEND = Object.entries(LABEL_COLORS).filter(([k]) => k !== 'File')

export default function GraphView() {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] })
  const [loading, setLoading]     = useState(true)
  const [error, setError]         = useState(null)
  const [tooltip, setTooltip]     = useState(null)
  const containerRef = useRef(null)
  const [dims, setDims] = useState({ w: 800, h: 600 })

  // Measure container
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => {
      setDims({ w: entry.contentRect.width, h: entry.contentRect.height })
    })
    ro.observe(el)
    setDims({ w: el.clientWidth, h: el.clientHeight })
    return () => ro.disconnect()
  }, [])

  // Fetch graph data
  useEffect(() => {
    setLoading(true)
    setError(null)
    fetch('/api/graph')
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(data => {
        const nodeSet = new Set(data.nodes.map(n => n.id))
        setGraphData({
          nodes: data.nodes.map(n => ({ ...n, color: LABEL_COLORS[n.label] || '#8b949e' })),
          links: data.edges
            .filter(e => nodeSet.has(e.source) && nodeSet.has(e.target))
            .map(e => ({ ...e, color: EDGE_COLORS[e.type] || 'rgba(139,148,158,0.4)' })),
        })
        setLoading(false)
      })
      .catch(err => { setError(err.message); setLoading(false) })
  }, [])

  const nodeCanvasObject = useCallback((node, ctx, globalScale) => {
    const r = Math.max(4, 8 - globalScale)
    ctx.beginPath()
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
    ctx.fillStyle = node.color
    ctx.fill()
    ctx.strokeStyle = 'rgba(255,255,255,0.15)'
    ctx.lineWidth = 0.8
    ctx.stroke()

    // Node name label — show from zoom 1.5+
    if (globalScale >= 1.5) {
      const fontSize = Math.max(3, 10 / globalScale)
      ctx.font = `${fontSize}px Inter, sans-serif`
      ctx.fillStyle = 'rgba(230,237,243,0.9)'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      ctx.fillText(node.name || '', node.x, node.y + r + 2 / globalScale)
    }
  }, [])

  // Edge label painter — renders the relationship type at the midpoint
  const linkCanvasObject = useCallback((link, ctx, globalScale) => {
    // Only draw labels when zoomed in enough to be readable
    if (globalScale < 1.5) return

    const start = link.source
    const end   = link.target
    if (typeof start !== 'object' || typeof end !== 'object') return

    const mx = (start.x + end.x) / 2
    const my = (start.y + end.y) / 2

    const fontSize = Math.max(3, 8 / globalScale)
    ctx.font = `500 ${fontSize}px Inter, sans-serif`

    const label  = link.type || ''
    const tw     = ctx.measureText(label).width
    const pad    = 2 / globalScale

    // Background pill so the label is readable over edges
    ctx.fillStyle = 'rgba(13,17,23,0.82)'
    ctx.beginPath()
    ctx.roundRect(
      mx - tw / 2 - pad,
      my - fontSize / 2 - pad,
      tw + pad * 2,
      fontSize + pad * 2,
      2 / globalScale,
    )
    ctx.fill()

    // Label text
    ctx.fillStyle = link.color || 'rgba(139,148,158,0.9)'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(label, mx, my)
  }, [])

  return (
    <div className="graph-view" ref={containerRef}>
      {/* Toolbar */}
      <div className="graph-toolbar">
        <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
          {loading ? 'Loading graph…'
            : error ? <span style={{ color: 'var(--red)' }}>{error}</span>
            : `${graphData.nodes.length} nodes · ${graphData.links.length} edges`}
        </span>
        <div className="graph-legend">
          {LEGEND.map(([label, color]) => (
            <div key={label} className="legend-item">
              <div className="legend-dot" style={{ background: color }} />
              {label}
            </div>
          ))}
        </div>
      </div>

      {/* Canvas */}
      <div className="graph-canvas">
        {!loading && graphData.nodes.length === 0 && (
          <div className="graph-empty">
            <Network size={48} style={{ opacity: 0.2 }} />
            <p>No nodes in the graph yet.<br />Ingest a repository to see the graph.</p>
          </div>
        )}
        {!loading && graphData.nodes.length > 0 && (
          <ForceGraph2D
            width={dims.w}
            height={dims.h - 53}
            graphData={graphData}
            nodeCanvasObject={nodeCanvasObject}
            nodeCanvasObjectMode={() => 'replace'}
            linkColor={link => link.color}
            linkWidth={1.2}
            linkDirectionalArrowLength={4}
            linkDirectionalArrowRelPos={1}
            linkCurvature={0.12}
            linkCanvasObject={linkCanvasObject}
            linkCanvasObjectMode={() => 'after'}
            backgroundColor="#0d1117"
            onNodeHover={(node, e) => {
              if (node) setTooltip({ node, x: e?.clientX ?? 0, y: e?.clientY ?? 0 })
              else setTooltip(null)
            }}
            cooldownTicks={80}
          />
        )}
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div
          className="node-tooltip"
          style={{ left: tooltip.x + 14, top: tooltip.y - 10 }}
        >
          <div className="node-tooltip-name">{tooltip.node.name}</div>
          <div className="node-tooltip-meta">
            {tooltip.node.label} · {tooltip.node.file || ''}
          </div>
        </div>
      )}
    </div>
  )
}
