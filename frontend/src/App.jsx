import { useState } from 'react'
import { MessageSquare, GitBranch, RefreshCw } from 'lucide-react'
import Sidebar from './components/Sidebar'
import ChatPanel from './components/ChatPanel'
import GraphView from './components/GraphView'
import './index.css'

const TABS = [
  { id: 'chat',  label: 'Chat',  Icon: MessageSquare },
  { id: 'graph', label: 'Graph', Icon: GitBranch },
]

export default function App() {
  const [tab, setTab] = useState('chat')
  const [graphRefresh, setGraphRefresh] = useState(0)

  return (
    <div className="app-shell">
      <Sidebar onIngestDone={() => setGraphRefresh(n => n + 1)} />

      <div className="main-panel">
        {/* Tab bar */}
        <div className="tab-bar">
          {TABS.map(({ id, label, Icon }) => (
            <button
              key={id}
              className={`tab ${tab === id ? 'active' : ''}`}
              onClick={() => setTab(id)}
            >
              <Icon size={14} />
              {label}
            </button>
          ))}
          {tab === 'graph' && (
            <button
              className="btn btn-ghost"
              style={{ marginLeft: 'auto', padding: '4px 10px', fontSize: 12 }}
              onClick={() => setGraphRefresh(n => n + 1)}
              title="Refresh graph"
            >
              <RefreshCw size={12} /> Refresh
            </button>
          )}
        </div>

        {/* Tab content */}
        <div className="tab-content">
          {tab === 'chat'  && <ChatPanel />}
          {tab === 'graph' && <GraphView key={graphRefresh} />}
        </div>
      </div>
    </div>
  )
}
