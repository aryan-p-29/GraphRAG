import { useState } from 'react'
import { Database, GitFork } from 'lucide-react'
import IngestPanel from './IngestPanel'

export default function Sidebar({ onIngestDone }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <GitFork size={22} />
          <h1>GraphRAG</h1>
        </div>
        <span className="sidebar-subtitle">Codebase Explorer</span>
      </div>

      <div className="sidebar-body">
        <p className="section-title">
          <Database size={12} /> Ingestion
        </p>
        <IngestPanel onDone={onIngestDone} />
      </div>
    </aside>
  )
}
