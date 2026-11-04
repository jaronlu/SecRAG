import { useState } from 'react'
import type { AnswerStatus, ChatMessage as ChatMessageType } from '../types'
import { formatAnswerHTML } from '../utils/markdown'

interface ChatMessageProps {
  message: ChatMessageType
}

// 对外业务终态的中文标签：状态标签必须有文字，不展示原始枚举
const ANSWER_STATUS_LABELS: Record<AnswerStatus, string> = {
  answered: '已回答',
  no_evidence: '未找到资料',
  permission_denied: '无权访问',
  needs_clarification: '需要补充信息',
  blocked: '已拦截',
}

// 失败终态的标签配色；answered 单独用绿色
const ANSWER_STATUS_STYLES: Record<AnswerStatus, string> = {
  answered: 'bg-emerald-100 text-emerald-700',
  no_evidence: 'bg-slate-100 text-slate-600',
  permission_denied: 'bg-amber-100 text-amber-700',
  needs_clarification: 'bg-amber-100 text-amber-700',
  blocked: 'bg-red-100 text-red-700',
}

// 无证据/无权限/拦截不得伪装成成功答案：不展示置信度评分
const STATUS_HIDE_CONFIDENCE: ReadonlySet<AnswerStatus> = new Set([
  'no_evidence',
  'permission_denied',
  'blocked',
])

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
  const answerStatus = message.answer_status
  const statusLabel = answerStatus ? ANSWER_STATUS_LABELS[answerStatus] : undefined
  const statusStyle = answerStatus ? ANSWER_STATUS_STYLES[answerStatus] : undefined
  const hideConfidence = answerStatus !== undefined && STATUS_HIDE_CONFIDENCE.has(answerStatus)

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

          {/* 回答内容：AI 消息按 markdown 渲染（formatAnswerHTML 内部先转义 HTML，
              输出中唯一 HTML 来自渲染器本身，流式光标也是常量字符串）。
              错误态用独立 alert 块纯文本渲染，不进 markdown/HTML 通道 */}
          {isUser ? (
            <div className="text-sm leading-relaxed whitespace-pre-wrap">
              {message.content}
              {message.streaming && <span className="cursor-blink">▋</span>}
            </div>
          ) : message.error ? (
            <div
              role="alert"
              className="text-sm leading-relaxed text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2"
            >
              {message.content}
            </div>
          ) : (
            <div
              className="text-sm leading-relaxed text-slate-800 md-answer"
              dangerouslySetInnerHTML={{
                __html:
                  formatAnswerHTML(message.content) +
                  (message.streaming ? '<span class="cursor-blink">▋</span>' : ''),
              }}
            />
          )}

          {/* 业务终态标签 + 置信度和合规：失败终态隐藏置信度评分 */}
          {!isUser && (statusLabel || (!hideConfidence && message.confidence)) && (
            <div className="flex gap-2 mt-2 text-[11px] flex-wrap">
              {statusLabel && (
                <span className={`px-2 py-0.5 rounded font-medium ${statusStyle}`}>
                  {statusLabel}
                </span>
              )}
              {!hideConfidence && message.confidence && (
                <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-600">
                  置信度: {message.confidence}
                </span>
              )}
              {!hideConfidence && message.compliance && (
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

        {/* 引用：只在 answered 且引用数组非空时展示；失败终态不带引用 */}
        {!isUser && message.answer_status === 'answered' && message.citations && message.citations.length > 0 && (
          <div className="mt-2 text-left">
            <button
              onClick={() => setShowCitations(!showCitations)}
              className="text-[11px] text-blue-600 hover:underline"
            >
              {showCitations ? '收起引用' : `查看 ${message.citations.length} 条引用`}
            </button>
            {showCitations && (
              <div className="mt-2 space-y-1">
                {message.citations.map((c, i) => {
                  // citation_id（cite_003）与 prompt 的来源序号一致；同一来源去重
                  // 时会跳号，展示编号必须取它的数字而不是数组位置，才能对上
                  // 正文里的 [来源3] 标注（issues.md 一.7）
                  const match = (c.citation_id ?? '').match(/cite_(\d+)/)
                  const sourceNumber = match ? Number(match[1]) : i + 1
                  return (
                    <div key={i} className="bg-slate-50 rounded p-2 text-[11px]">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium text-slate-700">[{sourceNumber}]</span>
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
                      {/* ISSUE-34：展示引用原文摘句，运营人员可就地核对引用内容；
                          后端 quote 为空时回退旧 content 字段保持兼容 */}
                      {(c.quote || c.content) && (
                        <div className="text-slate-500 mt-1 whitespace-pre-wrap line-clamp-3">
                          {c.quote || c.content}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
