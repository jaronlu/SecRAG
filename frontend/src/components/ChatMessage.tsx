import { useState } from 'react'
import type { ChatMessage as ChatMessageType } from '../types'

interface ChatMessageProps {
  message: ChatMessageType
}

const docTypeColors: Record<string, string> = {
  product: 'bg-blue-100 text-blue-700',
  regulation: 'bg-red-100 text-red-700',
  research: 'bg-green-100 text-green-700',
  faq: 'bg-amber-100 text-amber-700',
  unknown: 'bg-slate-100 text-slate-600',
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const [showCitations, setShowCitations] = useState(false)
  const isUser = message.role === 'user'

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* 头像 */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold shrink-0 ${
          isUser ? 'bg-blue-500 text-white' : 'bg-slate-700 text-white'
        }`}
      >
        {isUser ? '我' : 'AI'}
      </div>

      {/* 消息内容 */}
      <div className={`max-w-[80%] ${isUser ? 'text-right' : ''}`}>
        <div
          className={`inline-block text-left rounded-2xl px-4 py-3 ${
            isUser ? 'bg-blue-500 text-white rounded-tr-sm' : 'bg-white border border-slate-200 rounded-tl-sm'
          }`}
        >
          {/* 缓存标记 */}
          {message.cached && (
            <div className="text-[10px] text-green-600 mb-1 font-medium">⚡ 缓存命中</div>
          )}

          {/* 回答内容 */}
          <div className={`text-sm leading-relaxed whitespace-pre-wrap ${isUser ? '' : 'text-slate-800'}`}>
            {message.content}
            {message.streaming && <span className="cursor-blink">▋</span>}
          </div>

          {/* 置信度和合规 */}
          {!isUser && message.confidence && (
            <div className="flex gap-2 mt-2 text-[11px]">
              <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-600">
                置信度: {message.confidence}
              </span>
              {message.compliance && (
                <span
                  className={`px-2 py-0.5 rounded ${
                    message.compliance.passed ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                  }`}
                >
                  {message.compliance.passed ? '合规通过' : '合规未通过'}
                </span>
              )}
            </div>
          )}
        </div>

        {/* 引用 */}
        {!isUser && message.citations && message.citations.length > 0 && (
          <div className="mt-2 text-left">
            <button
              onClick={() => setShowCitations(!showCitations)}
              className="text-[11px] text-blue-600 hover:underline"
            >
              {showCitations ? '收起引用' : `查看 ${message.citations.length} 条引用`}
            </button>
            {showCitations && (
              <div className="mt-2 space-y-1">
                {message.citations.map((c, i) => (
                  <div key={i} className="bg-slate-50 rounded p-2 text-[11px]">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-medium text-slate-700">[{i + 1}]</span>
                      {c.doc_type && (
                        <span className={`px-1.5 py-0.5 rounded text-[10px] ${docTypeColors[c.doc_type] || docTypeColors.unknown}`}>
                          {c.doc_type}
                        </span>
                      )}
                      {c.score !== undefined && (
                        <span className="text-slate-400">相似度 {(c.score * 100).toFixed(0)}%</span>
                      )}
                    </div>
                    <div className="text-slate-600 truncate">{c.source}</div>
                    {c.content && <div className="text-slate-500 mt-1 line-clamp-2">{c.content}</div>}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
