import { useState, useRef, useEffect, useCallback } from 'react'
import ChatMessage from '../components/ChatMessage'
import { streamQuestion, createThread, getCurrentUser, listThreads, deleteThread, getThreadMessages } from '../api'
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

// ISSUE-33：与后端 AssistantQARequest.query max_length=500 对齐，
// 前端先拦截超长输入，避免提交后才收到 422
const MAX_QUERY_LENGTH = 500

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
  // BUG-2：正在查看历史会话（点击侧栏进入）与全新会话的空态不同——
  // 前者按 impl-07 §7 显示“暂无会话消息”，后者显示欢迎页
  const [isViewingHistory, setIsViewingHistory] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  // 加载会话列表
  const refreshThreads = useCallback(() => {
    listThreads()
      .then((data) => setThreads((data.threads || []).slice(0, 20)))
      .catch(() => {})
  }, [])

  useEffect(() => {
    refreshThreads()
  }, [refreshThreads])

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
      setIsViewingHistory(false)
      setMessages([])
      setThreads((prev) => [{ thread_id: thread.thread_id, title: '新对话' }, ...prev])
    } catch {
      setCurrentThreadId(undefined)
      setIsViewingHistory(false)
      setMessages([])
    }
  }

  // BUG-2：点击历史会话时从服务端加载消息列表（impl-07 §7「会话记录…展示
  // 消息列表」，GET /threads/{id}/messages 已存在但此前从未被调用）；
  // 加载失败必须显式提示，不得冒充“暂无会话消息”误导用户以为记录丢失
  const handleSelectThread = async (threadId: string) => {
    setCurrentThreadId(threadId)
    setIsViewingHistory(true)
    setHistoryLoading(true)
    setHistoryError(false)
    try {
      const data = await getThreadMessages(threadId)
      // list_messages 按 created_at DESC 返回，聊天视图恢复时间正序
      const history = [...(data.messages || [])].reverse().map((m) => ({
        id: m.message_id,
        role: m.role === 'user' ? ('user' as const) : ('assistant' as const),
        content: m.content,
        timestamp: m.created_at,
      }))
      setMessages(history)
    } catch {
      setMessages([])
      setHistoryError(true)
    } finally {
      setHistoryLoading(false)
    }
  }

  const handleDeleteThread = async (threadId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await deleteThread(threadId)
      setThreads((prev) => prev.filter((t) => t.thread_id !== threadId))
      if (currentThreadId === threadId) {
        setCurrentThreadId(undefined)
        setIsViewingHistory(false)
        setMessages([])
      }
    } catch {
      // 忽略
    }
  }

  const handleSend = async () => {
    // BUG-3：textarea 的 maxLength 只约束真实键入，程序化注入可绕过；
    // 发送前以 trim 后长度再做一道防线，与后端 query max_length=500 对齐
    if (!input.trim() || isStreaming || input.trim().length > MAX_QUERY_LENGTH) return

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

    // 传输或服务层失败的统一错误态：固定可操作文案，不拼底层异常。
    // ISSUE-33：可恢复的输入契约错误（422 等）用 api 层给出的具体文案，
    // 避免渲染"稍后重试"误导用户（重试永远无效）
    const markFailed = (message?: string) => {
      const text = message?.trim() || '请求未完成，请稍后重试。'
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId ? { ...m, content: text, streaming: false, error: true } : m,
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
            // ISSUE-35：本轮自动创建的会话（此前无 thread_id）立即进侧栏，
            // 否则进行中的会话在列表里不可见，只能点"新建对话"后才出现
            const isNewThread = !currentThreadId
            setCurrentThreadId(event.thread_id)
            if (isNewThread) {
              refreshThreads()
            }
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
    } catch (err) {
      // ISSUE-33：422 等输入契约错误携带具体文案，其余回退统一兜底文案
      markFailed(err instanceof Error ? err.message : undefined)
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
                onClick={() => void handleSelectThread(t.thread_id)}
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
            {messages.length > 0 ? (
              messages.map((m) => <ChatMessage key={m.id} message={m} />)
            ) : isViewingHistory ? (
              <div className="text-center py-20">
                <div className="text-slate-500 text-sm">
                  {historyLoading
                    ? '正在加载会话消息...'
                    : historyError
                      ? '会话消息加载失败，请稍后重试'
                      : '暂无会话消息'}
                </div>
              </div>
            ) : (
              <div className="text-center py-20">
                <div className="text-5xl mb-4">📊</div>
                <h2 className="text-xl font-bold text-slate-700 mb-2">证券智能问答</h2>
                <p className="text-slate-500 text-sm mb-6">
                  基于 RAG + Agent 的证券知识问答系统，支持产品咨询、法规查询、研报解读
                </p>
                <div className="flex flex-wrap gap-2 justify-center">
                  {/* ISSUE-37：推荐问题必须与当前演示语料对应（理财产品揭示书/
                    内部系统 FAQ/适当性制度），否则点击必然 no_evidence，
                    首触体验失真；语料更换时需同步调整 */}
                {[
                  '示例稳健增利理财产品的风险等级是多少？',
                  '如何重置内部系统的密码？',
                  '客户适当性管理制度的基本要求是什么？',
                ].map((q) => (
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
                maxLength={MAX_QUERY_LENGTH}
                disabled={isStreaming}
              />
              <button
                onClick={handleSend}
                disabled={isStreaming || !input.trim() || input.trim().length > MAX_QUERY_LENGTH}
                className="px-5 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white rounded-xl text-sm font-medium transition-colors"
              >
                {isStreaming ? '思考中...' : '发送'}
              </button>
            </div>
            <div className="flex justify-between items-center text-xs text-slate-400 mt-2">
              <span>SecRAG 证券智能问答 · 内容仅供参考，不构成投资建议</span>
              {/* ISSUE-33：与后端 query max_length=500 对齐；接近上限时提示 */}
              <span className={input.length >= MAX_QUERY_LENGTH ? 'text-amber-600' : ''}>
                {input.length}/{MAX_QUERY_LENGTH}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
