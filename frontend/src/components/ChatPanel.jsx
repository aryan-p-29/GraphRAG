import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, MessageSquare, Zap, Code2, GitMerge } from 'lucide-react'
import ChatMessage from './ChatMessage'

const SUGGESTIONS = [
  'What classes are defined and what do they do?',
  'Trace the execution path of the init() function.',
  'Which functions call the forecast() method?',
  'Show me the inheritance hierarchy.',
]

export default function ChatPanel() {
  const [messages, setMessages] = useState([])
  const [input, setInput]       = useState('')
  const [loading, setLoading]   = useState(false)
  const messagesEndRef = useRef(null)
  const textareaRef   = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const send = useCallback(async (text) => {
    const q = (text || input).trim()
    if (!q || loading) return

    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: q, ts: Date.now() }])
    setLoading(true)

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Query failed')
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        sources: data.sources || [],
        ts: Date.now(),
      }])
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `❌ **Error:** ${err.message}`,
        sources: [],
        ts: Date.now(),
      }])
    } finally {
      setLoading(false)
    }
  }, [input, loading])

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px'
  }, [input])

  return (
    <div className="chat-panel">
      <div className="messages">
        {messages.length === 0 && !loading && (
          <div className="empty-state">
            <MessageSquare size={48} />
            <h2>Ask about your codebase</h2>
            <p>Ingest a repository from the sidebar, then ask questions about its structure, execution paths, or relationships.</p>
            <div className="suggestions">
              {SUGGESTIONS.map(s => (
                <button key={s} className="suggestion-chip" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <ChatMessage key={i} message={msg} />
        ))}

        {loading && (
          <div className="message assistant">
            <div className="msg-avatar">
              <Zap size={15} />
            </div>
            <div className="msg-thinking">
              <div className="dot" /><div className="dot" /><div className="dot" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-bar">
        <div className="chat-input-row">
          <textarea
            ref={textareaRef}
            className="chat-textarea"
            placeholder="Ask about classes, functions, execution paths…  (Enter to send, Shift+Enter for newline)"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            rows={1}
          />
          <button
            className="send-btn"
            disabled={!input.trim() || loading}
            onClick={() => send()}
            title="Send"
          >
            <Send size={15} />
          </button>
        </div>
      </div>
    </div>
  )
}
