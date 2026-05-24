import { motion, useInView } from 'framer-motion'
import { useRef, useState, useEffect, useCallback } from 'react'

const RANKS = 4
const COLORS = ['#f43f5e', '#8b5cf6', '#06b6d4', '#f59e0b']
const GRID = 12
const ROWS_PER = GRID / RANKS

export default function MPIAnimation() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: false, margin: '-100px' })
  const [phase, setPhase] = useState('idle')
  const [progress, setProgress] = useState([0, 0, 0, 0])
  const [packets, setPackets] = useState([]) // {from, to, progress: 0-1}
  const tickRef = useRef(0)

  const reset = useCallback(() => {
    setPhase('idle')
    setProgress([0, 0, 0, 0])
    setPackets([])
    tickRef.current = 0
  }, [])

  useEffect(() => {
    if (!isInView) { reset(); return }

    const interval = setInterval(() => {
      tickRef.current++
      const t = tickRef.current

      if (t === 3) {
        setPhase('scatter')
        setPackets([
          { from: 0, to: 1, progress: 0 },
          { from: 0, to: 2, progress: 0 },
          { from: 0, to: 3, progress: 0 },
        ])
      }
      else if (t > 3 && t <= 18) {
        // Animate packets traveling from root to other ranks
        setPackets(prev => prev.map(p => ({ ...p, progress: Math.min(p.progress + 0.07, 1) })))
      }
      else if (t === 19) {
        setPhase('compute')
        setPackets([])
      }
      else if (t > 19 && t <= 59) {
        setProgress(prev => prev.map(p => Math.min(p + 2.5, 100)))
      }
      else if (t === 61) {
        setPhase('gather')
        setPackets([
          { from: 1, to: 0, progress: 0 },
          { from: 2, to: 0, progress: 0 },
          { from: 3, to: 0, progress: 0 },
        ])
      }
      else if (t > 61 && t <= 76) {
        setPackets(prev => prev.map(p => ({ ...p, progress: Math.min(p.progress + 0.07, 1) })))
      }
      else if (t === 77) {
        setPhase('done')
        setPackets([])
      }
      else if (t === 95) { reset() }
    }, 150)

    return () => clearInterval(interval)
  }, [isInView, reset])

  // Node positions (center of each rank node)
  const nodePositions = [
    { x: 140, y: 30 },  // Rank 0 (top center)
    { x: 260, y: 140 }, // Rank 1 (right)
    { x: 140, y: 250 }, // Rank 2 (bottom center)
    { x: 20, y: 140 },  // Rank 3 (left)
  ]

  return (
    <section ref={ref} className="min-h-screen flex items-center justify-center py-20 px-4">
      <div className="max-w-6xl w-full grid md:grid-cols-2 gap-12 items-center">
        <motion.div initial={{ opacity: 0, x: -40 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: false }}>
          <span className="text-xs uppercase tracking-widest text-cyan-400 font-semibold bg-cyan-500/10 px-3 py-1 rounded-full">Implementation 4 / 6</span>
          <h2 className="text-3xl md:text-4xl font-bold mt-4">
            <span className="text-cyan-400">MPI</span> Message Passing
          </h2>
          <p className="text-gray-400 mt-4 leading-relaxed">
            4 independent processes, each with its own memory. Root scatters row chunks, ranks compute in isolation, then results are gathered back.
          </p>

          {/* Phase bar */}
          <div className="mt-6 flex gap-2">
            {['scatter', 'compute', 'gather', 'done'].map(p => (
              <div key={p} className={`px-3 py-1.5 rounded-full text-[11px] font-semibold transition-all duration-300 ${
                phase === p ? 'bg-cyan-500/20 text-cyan-300 ring-1 ring-cyan-500/40 scale-105' : 'bg-gray-800/60 text-gray-600'
              }`}>
                {p === 'scatter' ? '📤 Scatter' : p === 'compute' ? '🧮 Compute' : p === 'gather' ? '📥 Gather' : '✅ Done'}
              </div>
            ))}
          </div>

          {/* Rank progress */}
          <div className="mt-6 space-y-2">
            {Array.from({ length: RANKS }).map((_, i) => (
              <div key={`rank-${i}`} className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[i] }} />
                <span className="text-[10px] text-gray-400 w-12 font-mono">Rank {i}</span>
                <div className="flex-1 h-3 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full rounded-full transition-all duration-300"
                    style={{ backgroundColor: COLORS[i], width: `${progress[i]}%` }} />
                </div>
                <span className="text-[10px] text-gray-600 w-10 text-right">{Math.round(progress[i])}%</span>
              </div>
            ))}
          </div>

          <div className="mt-6 grid grid-cols-3 gap-3">
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Processes</p>
              <p className="text-lg font-bold text-cyan-400">4</p>
            </div>
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Speedup</p>
              <p className="text-lg font-bold text-cyan-400">3.82×</p>
            </div>
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Time</p>
              <p className="text-lg font-bold text-cyan-400">20.86s</p>
            </div>
          </div>
        </motion.div>

        {/* Communication diagram with animated packets */}
        <motion.div initial={{ opacity: 0, x: 40 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: false }}
          className="flex flex-col items-center">
          <p className="text-[10px] text-gray-500 mb-2 uppercase tracking-wider">MPI Communication Topology</p>
          <div className="relative w-[300px] h-[300px] bg-gray-900/60 rounded-xl border border-gray-700/40">

            {/* Connection lines (drawn under nodes) */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 0 }}>
              {[1, 2, 3].map(i => (
                <line key={`line-${i}`}
                  x1={nodePositions[0].x + 30} y1={nodePositions[0].y + 30}
                  x2={nodePositions[i].x + 30} y2={nodePositions[i].y + 30}
                  stroke={phase === 'scatter' || phase === 'gather' ? '#22d3ee44' : '#374151'}
                  strokeWidth="2" strokeDasharray="6 4"
                  className="transition-all duration-500"
                />
              ))}
            </svg>

            {/* Rank nodes */}
            {nodePositions.map((pos, i) => (
              <div key={`node-${i}`}
                className="absolute w-[60px] h-[60px] rounded-xl flex flex-col items-center justify-center border-2 transition-all duration-500"
                style={{
                  left: pos.x, top: pos.y,
                  borderColor: COLORS[i],
                  backgroundColor: phase === 'compute' ? COLORS[i] + '25' : COLORS[i] + '10',
                  boxShadow: phase === 'compute' ? `0 0 20px ${COLORS[i]}33` : 'none',
                }}>
                <span className="text-[11px] font-bold" style={{ color: COLORS[i] }}>R{i}</span>
                <span className="text-[7px] text-gray-500">{i === 0 ? 'root' : `${ROWS_PER*i*180} rows`}</span>
                {/* Activity indicator */}
                {phase === 'compute' && (
                  <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-8 h-1 rounded-full overflow-hidden bg-gray-800">
                    <div className="h-full rounded-full transition-all duration-300" style={{ width: `${progress[i]}%`, backgroundColor: COLORS[i] }} />
                  </div>
                )}
              </div>
            ))}

            {/* Animated data packets traveling along paths */}
            {packets.map((pkt) => {
              const from = nodePositions[pkt.from]
              const to = nodePositions[pkt.to]
              const x = from.x + 30 + (to.x - from.x) * pkt.progress
              const y = from.y + 30 + (to.y - from.y) * pkt.progress
              const color = phase === 'scatter' ? COLORS[pkt.to] : COLORS[pkt.from]
              return (
                <div key={`pkt-${pkt.from}-${pkt.to}`}
                  className="absolute w-5 h-5 -translate-x-1/2 -translate-y-1/2 rounded-md flex items-center justify-center transition-all duration-150"
                  style={{
                    left: x, top: y,
                    backgroundColor: color + '55',
                    border: `1.5px solid ${color}`,
                    boxShadow: `0 0 8px ${color}66`,
                    opacity: pkt.progress < 0.95 ? 1 : 0,
                  }}>
                  <span className="text-[6px] font-bold" style={{ color }}>
                    {phase === 'scatter' ? '→' : '←'}
                  </span>
                </div>
              )
            })}

            {/* Center phase label */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none" style={{ zIndex: 5 }}>
              <div className="bg-gray-900/90 border border-gray-700/60 px-3 py-1.5 rounded-lg">
                <span className="text-[11px] font-mono text-cyan-300">
                  {phase === 'idle' && 'MPI_Init()'}
                  {phase === 'scatter' && 'MPI_Scatterv()'}
                  {phase === 'compute' && 'convolve()'}
                  {phase === 'gather' && 'MPI_Gatherv()'}
                  {phase === 'done' && 'MPI_Finalize() ✓'}
                </span>
              </div>
            </div>
          </div>

          {/* Image grid showing row distribution */}
          <div className="mt-4 w-[300px]">
            <p className="text-[9px] text-gray-500 mb-1">Image Row Distribution (2160 rows ÷ 4 ranks)</p>
            <div className="flex flex-col gap-[2px] rounded-lg overflow-hidden border border-gray-700/40">
              {Array.from({ length: RANKS }).map((_, r) => (
                <div key={`strip-${r}`} className="flex items-center">
                  <div className="relative h-5 flex-1 overflow-hidden transition-all duration-300"
                    style={{ backgroundColor: COLORS[r] + '15' }}>
                    {/* Fill bar showing computation progress */}
                    <div className="absolute inset-y-0 left-0 transition-all duration-300"
                      style={{ width: `${progress[r]}%`, backgroundColor: COLORS[r] + '40' }} />
                    <span className="absolute inset-0 flex items-center justify-center text-[8px] font-mono"
                      style={{ color: COLORS[r] }}>
                      Rank {r}: rows {r * 540}–{(r + 1) * 540 - 1}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-3 bg-cyan-500/10 border border-cyan-500/30 rounded-lg px-4 py-2 text-center">
            <p className="text-cyan-400 text-xs">Each rank has <span className="font-bold">separate address space</span> — no shared memory</p>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
