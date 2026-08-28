import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import {
  listDocuments,
  getDocumentStats,
  getDocumentChunks,
  deleteDocument,
  searchKnowledgeBase,
  getCacheStats,
} from '../api'
import type { DocumentInfo, DocumentStats, ChunkInfo, SearchResult, CacheStats } from '../types'

const docTypeColors: Record<string, string> = {
  product: 'bg-blue-100 text-blue-700',
  regulation: 'bg-red-100 text-red-700',
  research: 'bg-green-100 text-green-700',
  faq: 'bg-amber-100 text-amber-700',
  unknown: 'bg-slate-100 text-slate-600',
}

export default function AdminPage() {
  const [docs, setDocs] = useState<DocumentInfo[]>([])
  const [stats, setStats] = useState<DocumentStats | null>(null)
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedDoc, setSelectedDoc] = useState<DocumentInfo | null>(null)
  const [chunks, setChunks] = useState<ChunkInfo[]>([])
  const [chunksLoading, setChunksLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [searching, setSearching] = useState(false)

  const loadData = useCallback(async () => {
    setLoading(true)
    try {
      const [docsData, statsData, cacheData] = await Promise.all([
        listDocuments(),
        getDocumentStats(),
        getCacheStats().catch(() => null),
      ])
      setDocs(docsData.documents || [])
      setStats(statsData)
      setCacheStats(cacheData)
    } catch (err) {
      console.error('加载数据失败', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadData()
  }, [loadData])

  const handleViewChunks = async (doc: DocumentInfo) => {
    setSelectedDoc(doc)
    setChunksLoading(true)
    setChunks([])
    try {
      const data = await getDocumentChunks(doc.source)
      setChunks(data.chunks || [])
    } catch (err) {
      console.error('加载 chunk 失败', err)
    } finally {
      setChunksLoading(false)
    }
  }

  const handleDelete = async (doc: DocumentInfo) => {
    if (!confirm(`确定删除文档「${doc.title}」？此操作不可恢复。`)) return
    try {
      await deleteDocument(doc.source)
      setDocs((prev) => prev.filter((d) => d.source !== doc.source))
      loadData()
    } catch (err) {
      alert('删除失败: ' + (err instanceof Error ? err.message : '未知错误'))
    }
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    setSearching(true)
    setSearchResults([])
    try {
      const data = await searchKnowledgeBase(searchQuery.trim())
      setSearchResults(data.results || [])
    } catch (err) {
      console.error('搜索失败', err)
    } finally {
      setSearching(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-100">
      {/* 顶部栏 */}
      <header className="bg-slate-900 text-white px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-bold">SecRAG 管理后台</h1>
          <Link to="/" className="text-sm text-slate-400 hover:text-white">
            ← 返回对话
          </Link>
        </div>
        <button onClick={loadData} className="text-sm text-slate-400 hover:text-white">
          刷新
        </button>
      </header>

      <main className="max-w-6xl mx-auto p-6">
        {/* 统计卡片 */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <StatCard label="文档总数" value={stats?.total_documents ?? '-'} />
          <StatCard label="Chunk 总数" value={stats?.total_chunks ?? '-'} />
          <StatCard label="文档类型数" value={stats ? Object.keys(stats.by_doc_type).length : '-'} />
          <StatCard
            label="缓存命中率"
            value={cacheStats ? `${(cacheStats.hit_rate * 100).toFixed(1)}%` : '-'}
            sub={cacheStats ? `命中 ${cacheStats.total_hits} 次` : undefined}
          />
        </div>

        {/* 文档列表 */}
        <div className="bg-white rounded-xl shadow-sm mb-6">
          <div className="px-6 py-4 border-b border-slate-200">
            <h2 className="font-bold text-slate-800">文档列表</h2>
          </div>
          <div className="overflow-x-auto">
            {loading ? (
              <div className="text-center py-12 text-slate-400">加载中...</div>
            ) : docs.length === 0 ? (
              <div className="text-center py-12 text-slate-400">知识库为空，请先运行入库脚本</div>
            ) : (
              <table className="w-full">
                <thead>
                  <tr className="bg-slate-50">
                    <th className="text-left px-6 py-3 text-xs font-semibold text-slate-500 uppercase">文档</th>
                    <th className="text-left px-6 py-3 text-xs font-semibold text-slate-500 uppercase">类型</th>
                    <th className="text-left px-6 py-3 text-xs font-semibold text-slate-500 uppercase">Chunk 数</th>
                    <th className="text-right px-6 py-3 text-xs font-semibold text-slate-500 uppercase">操作</th>
                  </tr>
                </thead>
                <tbody>
                  {docs.map((doc) => (
                    <tr key={doc.source} className="border-t border-slate-100 hover:bg-slate-50">
                      <td className="px-6 py-3">
                        <div className="font-medium text-slate-800 text-sm">{doc.title}</div>
                        <div className="text-xs text-slate-400 truncate max-w-md">{doc.source}</div>
                      </td>
                      <td className="px-6 py-3">
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${docTypeColors[doc.doc_type] || docTypeColors.unknown}`}>
                          {doc.doc_type || 'unknown'}
                        </span>
                      </td>
                      <td className="px-6 py-3 text-sm text-slate-600">{doc.chunk_count}</td>
                      <td className="px-6 py-3 text-right">
                        <button
                          onClick={() => handleViewChunks(doc)}
                          className="text-xs text-blue-600 hover:underline mr-3"
                        >
                          查看
                        </button>
                        <button
                          onClick={() => handleDelete(doc)}
                          className="text-xs text-red-600 hover:underline"
                        >
                          删除
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* 语义搜索预览 */}
        <div className="bg-white rounded-xl shadow-sm">
          <div className="px-6 py-4 border-b border-slate-200">
            <h2 className="font-bold text-slate-800">语义搜索预览</h2>
            <p className="text-xs text-slate-500 mt-1">直接检索向量库，不经过 Agent，用于测试检索效果</p>
          </div>
          <div className="p-6">
            <div className="flex gap-2 mb-4">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="输入查询，测试检索效果..."
                className="flex-1 border border-slate-300 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-400"
              />
              <button
                onClick={handleSearch}
                disabled={searching || !searchQuery.trim()}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white rounded-lg text-sm font-medium"
              >
                {searching ? '搜索中...' : '搜索'}
              </button>
            </div>
            {searchResults.length > 0 && (
              <div className="space-y-3">
                {searchResults.map((r, i) => (
                  <div key={i} className="border border-slate-200 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-xs font-bold text-blue-600">#{i + 1}</span>
                      <span className="text-xs text-slate-500">相似度 {(r.score * 100).toFixed(1)}%</span>
                      <span className="text-xs text-slate-400 truncate">{String(r.metadata?.source || '')}</span>
                    </div>
                    <div className="text-sm text-slate-700 line-clamp-3">{r.content}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Chunk 详情弹窗 */}
      {selectedDoc && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={() => setSelectedDoc(null)}>
          <div className="bg-white rounded-xl w-full max-w-3xl max-h-[80vh] flex flex-col" onClick={(e) => e.stopPropagation()}>
            <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between">
              <h3 className="font-bold text-slate-800">{selectedDoc.title} - Chunk 详情</h3>
              <button onClick={() => setSelectedDoc(null)} className="text-slate-400 hover:text-slate-600 text-xl">
                ×
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-6">
              {chunksLoading ? (
                <div className="text-center py-12 text-slate-400">加载中...</div>
              ) : chunks.length === 0 ? (
                <div className="text-center py-12 text-slate-400">无 chunk 数据</div>
              ) : (
                <div className="space-y-4">
                  <div className="text-sm text-slate-500 mb-4">共 {chunks.length} 个 chunk</div>
                  {chunks.map((chunk) => (
                    <div key={chunk.chunk_id} className="border border-slate-200 rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-xs font-bold text-blue-600">Chunk #{chunk.chunk_index}</span>
                        <span className="text-xs text-slate-400">{chunk.chunk_id}</span>
                      </div>
                      <div className="text-sm text-slate-700 bg-slate-50 rounded p-3 whitespace-pre-wrap max-h-40 overflow-y-auto">
                        {chunk.content}
                      </div>
                      {chunk.metadata && Object.keys(chunk.metadata).length > 0 && (
                        <div className="mt-2 text-xs text-slate-400 flex flex-wrap gap-2">
                          {Object.entries(chunk.metadata).map(([k, v]) => (
                            <span key={k}>{k}: {String(v)}</span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-white rounded-xl shadow-sm p-4">
      <div className="text-xs text-slate-500 mb-1">{label}</div>
      <div className="text-2xl font-bold text-slate-800">{value}</div>
      {sub && <div className="text-xs text-slate-400 mt-1">{sub}</div>}
    </div>
  )
}
