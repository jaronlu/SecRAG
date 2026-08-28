// SecRAG 前端类型定义

export interface Citation {
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

// SSE 流式事件类型
export type StreamEventType = 'progress' | 'answer' | 'error' | 'done'

export interface StreamEvent {
  type: StreamEventType
  node?: string
  data?: string
  message?: string
}

// 流式进度节点
export const STREAM_NODES = [
  { key: 'query_understanding', label: '查询理解' },
  { key: 'retrieval_planning', label: '生成检索计划' },
  { key: 'retrieval_execution', label: '执行检索' },
  { key: 'result_filtering', label: '过滤结果' },
  { key: 'reasoning', label: '推理' },
  { key: 'verification', label: '验证' },
  { key: 'answer_organization', label: '组织回答' },
] as const
