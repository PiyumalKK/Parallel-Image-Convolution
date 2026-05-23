import { motion, useInView } from 'framer-motion'
import { useRef, useState, useEffect } from 'react'

const RANKS = 4
const OMP = 4
const RANK_COLORS = ['#f43f5e', '#a855f7', '#06b6d4', '#f59e0b']
const OMP_SHADES = [
  ['#fda4af', '#fb7185', '#f43f5e', '#e11d48'],
  ['#c4b5fd', '#a78bfa', '#8b5cf6', '#7c3aed'],
  ['#67e8f9', '#22d3ee', '#06b6d4', '#0891b2'],
  ['#fcd34d', '#fbbf24', '#f59e0b', '#d97706'],
]

export default function HybridAnimation() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: false, margin: '-100px' })
  const [phase, setPhase] = useState('idle')
  const [scatterProgress, setScatterProgress] = useState(0)
  const [ompProgress, setOmpProgress] = useState(
    Array.from({ length: RANKS }, () => Array.from({ length: OMP }, () => 0))
  )
  const [gatherProgress, setGatherProgress] = useState(0)

  useEffect(() => {
    if (!isInView) {
      setPhase('idle')
      setScatterProgress(0)
      setOmpProgress(Array.from({ length: RANKS }, () => Array.from({ length: OMP }, () => 0)))
      setGatherProgress(0)
      return
    }
    let tick = 0
    const interval = setInterval(() => {
      tick++
      if (tick <= 2) { setPhase('idle') }
      else if (tick <= 14) {
        setPhase('scatter')
        setScatterProgress(prev => Math.min(prev + 8.5, 100))
      }
      else if (tick === 15) {
        setPhase('fork')
      }
      else if (tick > 15 && tick <= 55) {
        setPhase('compute')
        setOmpProgress(prev =>
          prev.map((rank, r) =>
            rank.map((t, ti) => Math.min(t + 2.5 + (ti * 0.3), 100))
          )
        )
      }
      else if (tick === 56) { setPhase('join') }
      else if (tick > 57 && tick <= 69) {
        setPhase('gather')
        setGatherProgress(prev => Math.min(prev + 8.5, 100))
      }
      else if (tick === 70) { setPhase('done') }
      else if (tick === 82) {
        tick = 0
        setPhase('idle')
        setScatterProgress(0)
        setOmpProgress(Array.from({ length: RANKS }, () => Array.from({ length: OMP }, () => 0)))
        setGatherProgress(0)
      }
    }, 150)
    return () => clearInterval(interval)
  }, [isInView])

  return (
    <section ref={ref} className="min-h-screen flex items-center justify-center py-20 px-4">
      <div className="max-w-6xl w-full grid md:grid-cols-2 gap-12 items-center">
        <motion.div initial={{ opacity: 0, x: -40 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: false }}>
          <span className="text-xs uppercase tracking-widest text-pink-400 font-semibold bg-pink-500/10 px-3 py-1 rounded-full">Implementation 6 / 6</span>
          <h2 className="text-3xl md:text-4xl font-bold mt-4">
            <span className="text-pink-400">Hybrid</span> MPI + OpenMP
          </h2>
          <p className="text-gray-400 mt-4 leading-relaxed">
            Two-level parallelism: MPI distributes rows across 4 processes, then each process spawns 4 OpenMP threads internally. Best of both worlds.
          </p>

          {/* Two-level hierarchy diagram */}
          <div className="mt-6 bg-gray-800/60 rounded-xl p-4 border border-gray-700/50">
            <p className="text-[10px] uppercase text-gray-500 mb-3">Parallelism Hierarchy</p>
            <div className="flex flex-col items-center gap-2">
              {/* Level 1: Full image */}
              <div className="w-full h-6 rounded bg-gray-700/50 border border-gray-600/50 flex items-center justify-center">
                <span className="text-[9px] text-gray-400">Full Image (2160 × 3840)</span>
              </div>
              <span className="text-gray-600 text-[10px]">↓ MPI_Scatterv</span>
              {/* Level 2: MPI ranks */}
              <div className="w-full flex gap-1">
                {Array.from({ length: RANKS }).map((_, r) => (
                  <div key={`l2-${r}`} className="flex-1 h-6 rounded flex items-center justify-center transition-all duration-300"
                    style={{ backgroundColor: RANK_COLORS[r] + '25', border: `1px solid ${RANK_COLORS[r]}55` }}>
                    <span className="text-[8px] font-bold" style={{ color: RANK_COLORS[r] }}>R{r}</span>
                  </div>
                ))}
              </div>
              <span className="text-gray-600 text-[10px]">↓ #pragma omp parallel</span>
              {/* Level 3: OMP threads per rank */}
              <div className="w-full flex gap-1">
                {Array.from({ length: RANKS }).map((_, r) => (
                  <div key={`l3-${r}`} className="flex-1 flex gap-[2px]">
                    {Array.from({ length: OMP }).map((_, t) => (
                      <div key={`l3-${r}-${t}`} className="flex-1 h-4 rounded-sm"
                        style={{ backgroundColor: OMP_SHADES[r][t] + '40', border: `1px solid ${OMP_SHADES[r][t]}44` }} />
                    ))}
                  </div>
                ))}
              </div>
              <span className="text-[8px] text-gray-500">= 16 parallel workers</span>
            </div>
          </div>

          {/* Phase indicator */}
          <div className="mt-4 flex gap-1.5 flex-wrap">
            {['scatter', 'fork', 'compute', 'join', 'gather', 'done'].map(p => (
              <div key={p} className={`px-2 py-1 rounded-full text-[10px] font-semibold transition-all duration-300 ${
                phase === p ? 'bg-pink-500/20 text-pink-300 ring-1 ring-pink-500/40 scale-105' : 'bg-gray-800/60 text-gray-600'
              }`}>
                {p === 'scatter' ? '📤 Scatter' : p === 'fork' ? '🔀 Fork' : p === 'compute' ? '⚡ Compute' : p === 'join' ? '🔗 Join' : p === 'gather' ? '📥 Gather' : '✅ Done'}
              </div>
            ))}
          </div>

          <div className="mt-5 grid grid-cols-3 gap-3">
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Workers</p>
              <p className="text-lg font-bold text-pink-400">4×4 = 16</p>
            </div>
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Speedup</p>
              <p className="text-lg font-bold text-pink-400">18.1×</p>
            </div>
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Time</p>
              <p className="text-lg font-bold text-pink-400">4.49s</p>
            </div>
          </div>
        </motion.div>

        {/* Visual animation - right side */}
        <motion.div initial={{ opacity: 0, x: 40 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: false }}
          className="flex flex-col items-center gap-4">

          {/* Source image representation */}
          <div className="w-full max-w-[320px]">
            <p className="text-[9px] text-gray-500 mb-1 text-center uppercase tracking-wider">Live Execution View</p>

            {/* The image split into rank chunks, each showing OMP thread sub-rows */}
            <div className="rounded-xl border border-gray-700/40 overflow-hidden bg-gray-900/60">
              {Array.from({ length: RANKS }).map((_, r) => (
                <div key={`rank-vis-${r}`} className="relative">
                  {/* Rank header */}
                  <div className="flex items-center px-2 py-1 border-b transition-all duration-300"
                    style={{ borderColor: RANK_COLORS[r] + '44', backgroundColor: RANK_COLORS[r] + '08' }}>
                    <div className="w-2 h-2 rounded-full mr-1.5 transition-all duration-300"
                      style={{
                        backgroundColor: RANK_COLORS[r],
                        boxShadow: phase === 'compute' ? `0 0 6px ${RANK_COLORS[r]}` : 'none'
                      }} />
                    <span className="text-[9px] font-bold" style={{ color: RANK_COLORS[r] }}>
                      MPI Rank {r} {r === 0 && '(root)'}
                    </span>
                    <span className="text-[8px] text-gray-600 ml-auto font-mono">
                      rows {r * 540}–{(r + 1) * 540 - 1}
                    </span>
                  </div>

                  {/* OMP thread rows within this rank */}
                  <div className="flex">
                    {Array.from({ length: OMP }).map((_, t) => {
                      const filled = phase === 'compute' || phase === 'join' || phase === 'gather' || phase === 'done'
                        ? ompProgress[r][t] : 0
                      return (
                        <div key={`omp-${r}-${t}`} className="flex-1 h-8 relative border-r last:border-r-0"
                          style={{ borderColor: RANK_COLORS[r] + '15' }}>
                          {/* Thread fill */}
                          <div className="absolute inset-0 transition-all duration-300"
                            style={{
                              background: `linear-gradient(to right, ${OMP_SHADES[r][t]}55, ${OMP_SHADES[r][t]}22)`,
                              width: `${filled}%`,
                            }} />
                          {/* Thread label */}
                          <div className="absolute inset-0 flex items-center justify-center">
                            <span className="text-[7px] font-mono transition-all duration-300"
                              style={{ color: filled > 50 ? OMP_SHADES[r][t] : '#4b5563' }}>
                              T{t}
                            </span>
                          </div>
                        </div>
                      )
                    })}
                  </div>

                  {/* Scatter/gather overlay for this rank */}
                  {phase === 'scatter' && r > 0 && (
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      <div className="bg-gray-900/80 px-2 py-0.5 rounded text-[8px] font-mono transition-all duration-300"
                        style={{ color: RANK_COLORS[r], opacity: scatterProgress < r * 30 ? 0.3 : 1 }}>
                        ← receiving {540} rows
                      </div>
                    </div>
                  )}
                  {phase === 'gather' && r > 0 && (
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      <div className="bg-gray-900/80 px-2 py-0.5 rounded text-[8px] font-mono"
                        style={{ color: RANK_COLORS[r], opacity: gatherProgress > r * 25 ? 1 : 0.3 }}>
                        → sending results
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Communication status */}
          <div className="w-full max-w-[320px] bg-gray-900/60 rounded-lg border border-gray-700/40 p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[9px] uppercase text-gray-500">Communication</span>
              <span className="text-[9px] font-mono text-pink-400">
                {phase === 'idle' && 'MPI_Init()'}
                {phase === 'scatter' && 'MPI_Scatterv()'}
                {phase === 'fork' && '#pragma omp parallel'}
                {phase === 'compute' && 'convolve() × 16'}
                {phase === 'join' && 'omp barrier'}
                {phase === 'gather' && 'MPI_Gatherv()'}
                {phase === 'done' && 'MPI_Finalize() ✓'}
              </span>
            </div>

            {/* Visual flow: root → scatter arrows → ranks */}
            {(phase === 'scatter' || phase === 'gather') && (
              <div className="flex items-center gap-1 mt-1">
                <div className="w-8 h-6 rounded border flex items-center justify-center text-[8px] font-bold"
                  style={{ borderColor: RANK_COLORS[0], color: RANK_COLORS[0] }}>R0</div>
                <div className="flex-1 relative h-4">
                  {/* Animated arrow */}
                  <div className="absolute inset-y-0 left-0 flex items-center w-full">
                    <div className="h-[2px] rounded-full transition-all duration-300"
                      style={{
                        width: `${phase === 'scatter' ? scatterProgress : gatherProgress}%`,
                        backgroundColor: '#ec4899',
                        boxShadow: '0 0 6px #ec489966',
                      }} />
                  </div>
                  <span className="absolute inset-0 flex items-center justify-center text-[8px] text-pink-400">
                    {phase === 'scatter' ? '→ → →' : '← ← ←'}
                  </span>
                </div>
                <div className="flex gap-0.5">
                  {[1, 2, 3].map(i => (
                    <div key={`tgt-${i}`} className="w-6 h-6 rounded border flex items-center justify-center text-[7px] font-bold"
                      style={{ borderColor: RANK_COLORS[i], color: RANK_COLORS[i] }}>R{i}</div>
                  ))}
                </div>
              </div>
            )}

            {/* Fork visualization */}
            {(phase === 'fork' || phase === 'compute') && (
              <div className="mt-1 grid grid-cols-4 gap-1">
                {Array.from({ length: RANKS }).map((_, r) => (
                  <div key={`fork-${r}`} className="text-center">
                    <div className="text-[7px] font-bold mb-0.5" style={{ color: RANK_COLORS[r] }}>R{r}</div>
                    <div className="flex justify-center gap-[2px]">
                      {Array.from({ length: OMP }).map((_, t) => (
                        <div key={`dot-${r}-${t}`}
                          className="w-2 h-2 rounded-full transition-all duration-300"
                          style={{
                            backgroundColor: phase === 'compute' ? OMP_SHADES[r][t] : OMP_SHADES[r][t] + '44',
                            boxShadow: phase === 'compute' ? `0 0 4px ${OMP_SHADES[r][t]}88` : 'none',
                          }} />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="bg-pink-500/10 border border-pink-500/30 rounded-lg px-4 py-2 text-center">
            <p className="text-pink-400 text-xs">
              MPI for <span className="font-bold">inter-node</span> distribution + OpenMP for <span className="font-bold">intra-node</span> threading
            </p>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
