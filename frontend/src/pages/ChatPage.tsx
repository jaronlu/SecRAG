import { useState, useRef, useEffect, useCallback } from 'react'
import ChatMessage from '../components/ChatMessage'
import { streamQuestion, createThread, getCurrentUser, listThreads, deleteThread } from '../api'
import type { ChatMessage as ChatMessageType, CurrentUser, StreamEvent } from '../types'

// 角色取值与后端 TOKEN_USER_BINDINGS 一致；未知角色统一显示"已认证用户"，
// 不把原始枚举值当文案展示
const ROLE_LABELS: Record<string, string> = {
  advisor: '投资顾问',
  institutional_sales: '机构销售',
  compliance: '合规',
  operations: '运营支持',
  technical: '技术支持',
}

function roleLabelOf(user: CurrentUser | null): string {
  if (!user) return '已认证用户'
  return ROLE_LABELS[user.role] ?? '已认证用户'
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessageType[]>([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [threads, setThreads] = useState<{ thread_id: string; title: string }[]>([])
  const [currentThreadId, setCurrentThreadId] = useState<string | undefined>()
  const [currentUser, setCurrentUser] = useState<CurrentUser | null>(null)
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

  // 身份由服务端从 Bearer token 派生，页面不提供角色选择或 Token 编辑
  useEffect(() => {
    getCurrentUser()
      .then(setCurrentUser)
      .catch(() => {})
  }, [])

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

    const assistantId = (Date.now() + 1).toString()

    // 唯一主流程：始终走 SSE 流式；进度/内部节点不外显
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: 'assistant', content: '', streaming: true },
    ])

    // 传输或服务层失败的统一错误态：固定可操作文案，不拼底层异常
    const markFailed = () => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: '请求未完成，请稍后重试。', streaming: false, error: true }
            : m,
        ),
      )
    }

    try {
      await streamQuestion(input.trim(), currentThreadId, (event: StreamEvent) => {
        if (event.type === 'answer_delta' && event.delta) {
          // ISSUE-9：真流式——reason 节点 token 增量直接上屏
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + event.delta } : m,
            ),
          )
        } else if (event.type === 'answer') {
          // 后端 answer 事件承载完整终态：answer 文本 + 引用 + 置信度 + 业务终态
          const text = event.answer ?? ''
          const citations = event.citations
          const confidence = event.confidence
          const answerStatus = event.answer_status
          // 自动创建会话后，把 thread_id 回写，后续问题沿用同一会话
          if (event.thread_id) {
            setCurrentThreadId(event.thread_id)
          }
          // 终态文本为权威内容：覆盖已流出的增量（未走 LLM 流式的
          // 拒答/澄清/拦截文案只经 answer 事件到达）
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: text, citations, confidence, answer_status: answerStatus, streaming: false }
                : m,
            ),
          )
        } else if (event.type === 'error') {
          markFailed()
        }
        // progress 事件后端可能仍发送：前端不消费，内部节点不外显
      })
    } catch {
      markFailed()
    }

    // 流式结束后，标记完成（错误态已带终稿文本，不再覆盖）
    setMessages((prev) =>
      prev.map((m) => (m.id === assistantId ? { ...m, streaming: false } : m)),
    )
    setIsStreaming(false)
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
          <h1 className="text-lg font-bold mb-3">SecRAG</h1>
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

        {/* 身份信息：由服务端派生，页面不提供编辑入口 */}
        <div className="p-3 border-t border-slate-700">
          <div className="text-xs text-slate-400">
            当前身份：{roleLabelOf(currentUser)}
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
              {roleLabelOf(currentUser)}
            </h2>
          </div>
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
    </div>
  )
}
