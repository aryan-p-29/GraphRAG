import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { Zap, User } from 'lucide-react'

function fmt(ts) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const CODE = {
  code({ node, inline, className, children, ...props }) {
    const match = /language-(\w+)/.exec(className || '')
    return !inline && match ? (
      <SyntaxHighlighter
        style={vscDarkPlus}
        language={match[1]}
        PreTag="div"
        customStyle={{ borderRadius: 6, fontSize: 12.5, margin: '8px 0', background: 'var(--bg-base)' }}
        {...props}
      >
        {String(children).replace(/\n$/, '')}
      </SyntaxHighlighter>
    ) : (
      <code className={className} {...props}>{children}</code>
    )
  }
}

export default function ChatMessage({ message }) {
  const isUser = message.role === 'user'

  // Clean up LLM hallucinations of internal LlamaIndex UUIDs
  // This cleans both the message text and the source chips
  const cleanContent = isUser 
    ? message.content 
    : (message.content || '').replace(/[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}/gi, '')

  const cleanSources = (message.sources || []).filter(s => 
    !/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i.test(s)
  )

  return (
    <div className={`message ${message.role}`}>
      <div className="msg-avatar">
        {isUser ? <User size={15} /> : <Zap size={15} />}
      </div>
      <div className="msg-body">
        <div className="msg-bubble">
          {isUser ? (
            <p style={{ margin: 0 }}>{cleanContent}</p>
          ) : (
            <ReactMarkdown components={CODE}>{cleanContent}</ReactMarkdown>
          )}
        </div>

        {/* Source chips */}
        {!isUser && cleanSources.length > 0 && (
          <div className="sources">
            {cleanSources.slice(0, 8).map((s, i) => (
              <span key={i} className="source-chip" title={s}>
                {s.split('::').pop()}
              </span>
            ))}
          </div>
        )}

        <span className="msg-time">{fmt(message.ts)}</span>
      </div>
    </div>
  )
}

