// SecRAG API 客户端

import type {
  AssistantQAResponse,
  ConversationThread,
  DocumentInfo,
  DocumentStats,
  ChunkInfo,
  SearchResult,
  CacheStats,
  StreamEvent,
} from './types'

const API_BASE = ''

function getToken(): string {
  return localStorage.getItem('secrag_token') || 'demo-advisor'
}

function authHeaders(): Record<string, string> {
  return {
    Authorization: `Bearer ${getToken()}`,
    'Content-Type': 'application/json',
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(API_BASE + path, {
    ...options,
    headers: { ...authHeaders(), ...options?.headers },
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`HTTP ${res.status}: ${text || res.statusText}`)
  }
  return res.json()
}

// ============ 对话相关 ============

export async function createThread(title: string): Promise<ConversationThread> {
  return request<ConversationThread>('/v1/assistant/threads', {
    method: 'POST',
    body: JSON.stringify({ title }),
  })
}

export async function listThreads(): Promise<{ threads: ConversationThread[] }> {
  return request('/v1/assistant/threads')
}

export async function getThreadMessages(threadId: string): Promise<{ messages: unknown[] }> {
  return request(`/v1/assistant/threads/${threadId}/messages`)
}

export async function deleteThread(threadId: string): Promise<void> {
  await request(`/v1/assistant/threads/${threadId}`, { method: 'DELETE' })
}

export async function askQuestion(
  query: string,
  threadId?: string,
): Promise<AssistantQAResponse> {
  return request<AssistantQAResponse>('/v1/assistant/qa', {
    method: 'POST',
    body: JSON.stringify({ query, thread_id: threadId }),
  })
}

// ============ 流式输出 ============

export async function streamQuestion(
  query: string,
  threadId: string | undefined,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  const res = await fetch(API_BASE + '/v1/assistant/qa/stream', {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ query, thread_id: threadId }),
  })

  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`)
  }

  const reader = res.body?.getReader()
  if (!reader) throw new Error('No response body')

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const dataStr = line.slice(6).trim()
      if (!dataStr || dataStr === '[DONE]') continue
      try {
        const event = JSON.parse(dataStr) as StreamEvent
        onEvent(event)
      } catch {
        // 忽略解析错误
      }
    }
  }
}

// ============ 知识库管理 ============

export async function listDocuments(docType?: string): Promise<{ total: number; documents: DocumentInfo[] }> {
  const params = docType ? `?doc_type=${encodeURIComponent(docType)}` : ''
  return request(`/v1/admin/documents${params}`)
}

export async function getDocumentStats(): Promise<DocumentStats> {
  return request('/v1/admin/documents/stats')
}

export async function getDocumentChunks(source: string): Promise<{ source: string; total: number; chunks: ChunkInfo[] }> {
  return request(`/v1/admin/documents/chunks?source=${encodeURIComponent(source)}&limit=100`)
}

export async function deleteDocument(source: string): Promise<{ source: string; deleted_chunks: number }> {
  return request(`/v1/admin/documents?source=${encodeURIComponent(source)}`, { method: 'DELETE' })
}

export async function searchKnowledgeBase(query: string, topK = 5): Promise<{ query: string; results: SearchResult[] }> {
  return request(`/v1/admin/documents/search?query=${encodeURIComponent(query)}&top_k=${topK}`)
}

// ============ 缓存 ============

export async function getCacheStats(): Promise<CacheStats> {
  return request('/v1/admin/cache/stats')
}

export async function clearCache(expiredOnly = false): Promise<{ cleared: number; mode: string }> {
  return request(`/v1/admin/cache/clear?clear_expired_only=${expiredOnly}`, { method: 'POST' })
}

// ============ 健康检查 ============

export async function getHealth(): Promise<Record<string, unknown>> {
  const res = await fetch(API_BASE + '/health')
  return res.json()
}
