// SecRAG 前端类型定义

export interface Citation {
  // cite_00N，数字与 prompt 的来源序号一致（issues.md 一.7）
  citation_id?: string
  // ISSUE-34：后端只回文件名，不再暴露服务端绝对路径
  source: string
  title?: string
  doc_type?: string
  // 引用原文摘句（CitationExtractor 产出，支撑答案事实的句子）
  quote?: string
  content?: string
  score?: number
  [key: string]: unknown
}

export interface ComplianceResult {
  passed: boolean
  reason?: string
  [key: string]: unknown
}

export interface AssistantQAResponse {
  thread_id: string
  turn_id: string
  answer: string
  citations: Citation[]
  confidence: string
  compliance: ComplianceResult
  cached?: boolean
  cache_similarity?: number
}

export interface ConversationThread {
  thread_id: string
  title: string
  created_at: string
  updated_at: string
  [key: string]: unknown
}

// 当前认证上下文：角色由 Bearer token 服务端派生，前端不提供选择或编辑入口
export interface CurrentUser {
  role: string
  department: string
  product_profile: 'knowledge_qa' | 'full'
}

// 对外业务终态（与后端 Product Contract 的五个 answer_status 一一对应）
export type AnswerStatus =
  | 'answered'
  | 'no_evidence'
  | 'permission_denied'
  | 'needs_clarification'
  | 'blocked'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  confidence?: string
  compliance?: ComplianceResult
  timestamp?: string
  streaming?: boolean
  cached?: boolean
  // 传输/服务层失败标记：独立错误态渲染，不把异常正文拼进回答
  error?: boolean
  answer_status?: AnswerStatus
}

export interface DocTypeBadge {
  type: string
  label: string
  className: string
}

export interface DocumentInfo {
  source: string
  doc_id: string
  doc_type: string
  title: string
  chunk_count: number
}

export interface DocumentStats {
  total_chunks: number
  total_documents: number
  by_doc_type: Record<string, number>
  by_doc_type_chunks: Record<string, number>
}

export interface ChunkInfo {
  chunk_id: string
  chunk_index: number
  content: string
  metadata: Record<string, unknown>
}

export interface SearchResult {
  content: string
  metadata: Record<string, unknown>
  score: number
}

export interface CacheStats {
  total_entries: number
  active_entries: number
  expired_entries: number
  total_hits: number
  hit_rate: number
  threshold: number
  ttl_seconds: number
  enabled: boolean
}

// SSE 流式事件类型（与后端 assistant_qa_stream 的事件协议一致：
// event 名与 JSON 内 type 字段相同，issues.md 一.3）
export type StreamEventType = 'progress' | 'answer_delta' | 'answer' | 'error' | 'done'

export interface StreamEvent {
  type: StreamEventType
  node?: string
  // answer_delta 携带 reason 节点 LLM token 级增量（ISSUE-9），先于 answer 终态流出
  delta?: string
  answer?: string
  citations?: Citation[]
  confidence?: string
  // 对外业务终态（Product Contract 五态），随 answer 终态事件到达
  answer_status?: AnswerStatus
  thread_id?: string
  turn_id?: string
  detail?: string
}

// 流式进度节点名与 SSE progress 事件对产品用户不外显：问答页只消费
// answer_delta/answer/error/done，progress 事件到达时忽略。
// 后端节点名集合（CLIENT_PROGRESS_NODES）仅供审计/观测使用，
// 契约由 tests/test_stream_progress_contract.py 守护
