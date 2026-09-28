// SecRAG 前端类型定义

export interface Citation {
  // cite_00N，数字与 prompt 的来源序号一致（issues.md 一.7）
  citation_id?: string
  source: string
  title?: string
  doc_type?: string
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
export type StreamEventType = 'progress' | 'answer' | 'error' | 'done'

export interface StreamEvent {
  type: StreamEventType
  node?: string
  answer?: string
  citations?: Citation[]
  confidence?: string
  thread_id?: string
  turn_id?: string
  detail?: string
}

// 流式进度节点。key 必须与后端图节点注册名一致（src/agents/graph.py 的
// CLIENT_PROGRESS_NODES），SSE progress 事件携带的就是这些名字；
// 两侧契约由 tests/test_stream_progress_contract.py 守护
export const STREAM_NODES = [
  { key: 'query_understand', label: '查询理解' },
  { key: 'planner', label: '生成检索计划' },
  { key: 'retrieve', label: '执行检索' },
  { key: 'grade_and_filter', label: '过滤结果' },
  { key: 'reason', label: '推理' },
  { key: 'verify', label: '验证' },
  { key: 'compose', label: '组织回答' },
] as const
