import { useState, useRef, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import ChatMessage from '../components/ChatMessage'
import StreamingProgress from '../components/StreamingProgress'
import { askQuestion, streamQuestion, createThread, listThreads, deleteThread } from '../api'
import type { ChatMessage as ChatMessageType, StreamEvent } from '../types'

// 角色取值必须与后端 TOKEN_USER_BINDINGS 一致（issues.md 一.3）
const ROLES = [
  { value: 'demo-advisor', label: '投资顾问', desc: '可查看产品和研报' },
  { value: 'demo-sales', label: '机构销售', desc: '可查看研报和市场信息' },
  { value: 'demo-compliance', label: '合规', desc: '可查看法规和制度' },
  { value: 'demo-ops', label: '运营支持', desc: '可查看FAQ和流程' },
  { value: 'demo-tech', label: '技术支持', desc: '可查看技术文档' },
]

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessageType[]>([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamEnabled, setStreamEnabled] = useState(true)
  const [currentNode, setCurrentNode] = useState<string | null>(null)
  const [threads, setThreads] = useState<{ thread_id: string; title: string }[]>([])
  const [currentThreadId, setCurrentThreadId] = useState<string | undefined>()
  const [token, setToken] = useState(localStorage.getItem('secrag_token') || 'demo-advisor')
  const [showSettings, setShowSettings] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  // 加载会话列表
  useEffect(() => {
    listThreads()
      .then((data) => setThreads((data.threads || []).slice(0, 20)))
      .catch(() => {})
  }, [])

  const handleTokenChange = (newToken: string) => {
    setToken(newToken)
    localStorage.setItem('secrag_token', newToken)
  }

  const handleNewThread = async () => {
    try {
      const thread = await createThread('新对话')
      setCurrentThreadId(thread.thread_id)
      setMessages([])
      setThreads((prev) => [{ thread_id: thread.thread_id, title: '新对话' }, ...prev])
    } catch {
      setCurrentThreadId(undefined)
      setMessages([])
    }
  }

  const handleDeleteThread = async (threadId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await deleteThread(threadId)
      setThreads((prev) => prev.filter((t) => t.thread_id !== threadId))
      if (currentThreadId === threadId) {
        setCurrentThreadId(undefined)
        setMessages([])
      }
    } catch {
      // 忽略
    }
  }

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return

    const userMessage: ChatMessageType = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
    }
    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsStreaming(true)
    setCurrentNode(null)

    const assistantId = (Date.now() + 1).toString()

    if (streamEnabled) {
      // 流式输出
      setMessages((prev) => [
        ...prev,
        { id: assistantId, role: 'assistant', content: '', streaming: true },
      ])

      try {
        await streamQuestion(input.trim(), currentThreadId, (event: StreamEvent) => {
          if (event.type === 'progress' && event.node) {
            setCurrentNode(event.node)
          } else if (event.type === 'answer') {
            // 后端 answer 事件承载完整终态：answer 文本 + 引用 + 置信度
            const text = event.answer ?? ''
            const citations = event.citations
            const confidence = event.confidence
            // 自动创建会话后，把 thread_id 回写，后续问题沿用同一会话
            if (event.thread_id) {
              setCurrentThreadId(event.thread_id)
            }
            // 打字机效果
            let i = 0
            const typeInterval = setInterval(() => {
              if (i < text.length) {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, content: text.slice(0, i + 1), citations, confidence }
                      : m,
                  ),
                )
                i++
              } else {
                clearInterval(typeInterval)
              }
            }, 10)
          } else if (event.type === 'error') {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: `错误: ${event.detail ?? '未知错误'}`, streaming: false }
                  : m,
              ),
            )
          } else if (event.type === 'done') {
            setCurrentNode(null)
          }
        })
      } catch (err) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: `错误: ${err instanceof Error ? err.message : '未知错误'}`, streaming: false }
              : m,
          ),
        )
      }

      // 流式结束后，标记完成
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, streaming: false } : m)),
      )
    } else {
      // 同步输出
      try {
        const result = await askQuestion(input.trim(), currentThreadId)
        // 未显式新建会话时，问答会自动创建会话——回写 thread_id 保证连续对话
        if (result.thread_id) {
          setCurrentThreadId(result.thread_id)
        }
        setMessages((prev) => [
          ...prev,
          {
            id: assistantId,
            role: 'assistant',
            content: result.answer,
            citations: result.citations,
            confidence: result.confidence,
            compliance: result.compliance,
            cached: result.cached,
          },
        ])
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          {
            id: assistantId,
            role: 'assistant',
            content: `错误: ${err instanceof Error ? err.message : '未知错误'}`,
          },
        ])
      }
    }

    setIsStreaming(false)
    setCurrentNode(null)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex h-screen bg-slate-100">
      {/* 侧边栏 */}
      <div
        className={`${sidebarOpen ? 'w-64' : 'w-0'} bg-slate-900 text-white flex flex-col transition-all duration-200 overflow-hidden`}
      >
        <div className="p-4 border-b border-slate-700">
          <div className="flex items-center justify-between mb-3">
            <h1 className="text-lg font-bold">SecRAG</h1>
            <Link to="/admin" className="text-xs text-slate-400 hover:text-white">
              管理后台
            </Link>
          </div>
          <button
            onClick={handleNewThread}
            className="w-full py-2 px-3 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium transition-colors"
          >
            + 新建对话
          </button>
        </div>

        {/* 会话列表 */}
        <div className="flex-1 overflow-y-auto p-2">
          {threads.length === 0 ? (
            <div className="text-center text-slate-500 text-xs py-8">暂无会话</div>
          ) : (
            threads.map((t) => (
              <div
                key={t.thread_id}
                onClick={() => {
                  setCurrentThreadId(t.thread_id)
                  setMessages([])
                }}
                className={`group flex items-center justify-between p-2 rounded-lg cursor-pointer text-sm mb-1 ${
                  currentThreadId === t.thread_id ? 'bg-slate-700' : 'hover:bg-slate-800'
                }`}
              >
                <span className="truncate flex-1">{t.title || '新对话'}</span>
                <button
                  onClick={(e) => handleDeleteThread(t.thread_id, e)}
                  className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-400 ml-2"
                >
                  ×
                </button>
              </div>
            ))
          )}
        </div>

        {/* 角色和设置 */}
        <div className="p-3 border-t border-slate-700">
          <div className="mb-2">
            <label className="text-xs text-slate-400 block mb-1">当前角色</label>
            <select
              value={token}
              onChange={(e) => handleTokenChange(e.target.value)}
              className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1.5 text-sm"
            >
              {ROLES.map((r) => (
                <option key={r.value} value={r.value}>
                  {r.label}
                </option>
              ))}
            </select>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">流式输出</span>
            <button
              onClick={() => setStreamEnabled(!streamEnabled)}
              className={`w-10 h-5 rounded-full transition-colors ${streamEnabled ? 'bg-blue-500' : 'bg-slate-600'}`}
            >
              <div
                className={`w-4 h-4 bg-white rounded-full transition-transform ${streamEnabled ? 'translate-x-5' : 'translate-x-0.5'}`}
              />
            </button>
          </div>
        </div>
      </div>

      {/* 主区域 */}
      <div className="flex-1 flex flex-col">
        {/* 顶部栏 */}
        <div className="bg-white border-b border-slate-200 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-slate-500 hover:text-slate-700"
            >
              ☰
            </button>
            <h2 className="font-medium text-slate-700">
              {ROLES.find((r) => r.value === token)?.label || '对话'}
            </h2>
          </div>
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="text-slate-400 hover:text-slate-600 text-sm"
          >
            ⚙
          </button>
        </div>

        {/* 消息列表 */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.length === 0 ? (
              <div className="text-center py-20">
                <div className="text-5xl mb-4">📊</div>
                <h2 className="text-xl font-bold text-slate-700 mb-2">证券智能问答</h2>
                <p className="text-slate-500 text-sm mb-6">
                  基于 RAG + Agent 的证券知识问答系统，支持产品咨询、法规查询、研报解读
                </p>
                <div className="flex flex-wrap gap-2 justify-center">
                  {['货币基金风险等级', '股票质押式回购新规', '什么是量化交易'].map((q) => (
                    <button
                      key={q}
                      onClick={() => setInput(q)}
                      className="px-3 py-1.5 bg-white border border-slate-200 rounded-full text-sm text-slate-600 hover:border-blue-400 hover:text-blue-600 transition-colors"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((m) => <ChatMessage key={m.id} message={m} />)
            )}

            {/* 流式进度 */}
            {isStreaming && streamEnabled && messages.length > 0 && (
              <div className="max-w-[80%]">
                <StreamingProgress currentNode={currentNode} completed={false} />
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* 输入框 */}
        <div className="bg-white border-t border-slate-200 p-4">
          <div className="max-w-3xl mx-auto">
            <div className="flex gap-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="输入你的问题...（Enter 发送，Shift+Enter 换行）"
                className="flex-1 border border-slate-300 rounded-xl px-4 py-3 text-sm resize-none focus:outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
                rows={1}
                disabled={isStreaming}
              />
              <button
                onClick={handleSend}
                disabled={isStreaming || !input.trim()}
                className="px-5 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white rounded-xl text-sm font-medium transition-colors"
              >
                {isStreaming ? '思考中...' : '发送'}
              </button>
            </div>
            <div className="text-center text-xs text-slate-400 mt-2">
              SecRAG 证券智能问答 · 内容仅供参考，不构成投资建议
            </div>
          </div>
        </div>
      </div>

      {/* 设置弹窗 */}
      {showSettings && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowSettings(false)}>
          <div className="bg-white rounded-xl p-6 w-96" onClick={(e) => e.stopPropagation()}>
            <h3 className="font-bold text-lg mb-4">设置</h3>
            <div className="space-y-4">
              <div>
                <label className="text-sm text-slate-600 block mb-1">API Token</label>
                <input
                  type="text"
                  value={token}
                  onChange={(e) => handleTokenChange(e.target.value)}
                  className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm"
                />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-600">流式输出</span>
                <button
                  onClick={() => setStreamEnabled(!streamEnabled)}
                  className={`w-10 h-5 rounded-full transition-colors ${streamEnabled ? 'bg-blue-500' : 'bg-slate-300'}`}
                >
                  <div className={`w-4 h-4 bg-white rounded-full transition-transform ${streamEnabled ? 'translate-x-5' : 'translate-x-0.5'}`} />
                </button>
              </div>
            </div>
            <button onClick={() => setShowSettings(false)} className="w-full mt-6 py-2 bg-slate-100 hover:bg-slate-200 rounded-lg text-sm">
              关闭
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
