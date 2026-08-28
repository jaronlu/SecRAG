import { STREAM_NODES } from '../types'

interface StreamingProgressProps {
  currentNode: string | null
  completed: boolean
}

export default function StreamingProgress({ currentNode, completed }: StreamingProgressProps) {
  const currentIndex = STREAM_NODES.findIndex((n) => n.key === currentNode)

  return (
    <div className="bg-slate-50 rounded-lg p-3 mb-3">
      <div className="text-xs text-slate-500 mb-2 font-medium">AI 思考中...</div>
      <div className="flex items-center gap-1 flex-wrap">
        {STREAM_NODES.map((node, idx) => {
          const isDone = completed || idx < currentIndex
          const isActive = !completed && idx === currentIndex
          return (
            <div key={node.key} className="flex items-center gap-1">
              <div
                className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold transition-all ${
                  isDone
                    ? 'bg-green-500 text-white'
                    : isActive
                      ? 'bg-blue-500 text-white progress-active'
                      : 'bg-slate-200 text-slate-400'
                }`}
              >
                {isDone ? '✓' : idx + 1}
              </div>
              <span
                className={`text-[11px] ${
                  isDone ? 'text-green-600' : isActive ? 'text-blue-600 font-medium' : 'text-slate-400'
                }`}
              >
                {node.label}
              </span>
              {idx < STREAM_NODES.length - 1 && (
                <div className={`w-3 h-0.5 ${isDone ? 'bg-green-400' : 'bg-slate-200'}`} />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
