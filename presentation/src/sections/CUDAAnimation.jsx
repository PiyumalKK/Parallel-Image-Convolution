import { motion, useInView } from 'framer-motion'
import { useRef, useState, useEffect } from 'react'

const BLOCKS_X = 8
const BLOCKS_Y = 4
const TOTAL_BLOCKS = BLOCKS_X * BLOCKS_Y

export default function CUDAAnimation() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: false, margin: '-100px' })
  const [activeBlock, setActiveBlock] = useState(-1)
  const [done, setDone] = useState(new Set())
  const [transferPhase, setTransferPhase] = useState('idle')

  useEffect(() => {
    if (!isInView) { setActiveBlock(-1); setDone(new Set()); setTransferPhase('idle'); return }
    let tick = 0
    const interval = setInterval(() => {
      tick++
      if (tick <= 3) { setTransferPhase('h2d') }
      else if (tick <= 3 + TOTAL_BLOCKS) {
        setTransferPhase('compute')
        const blk = tick - 4
        setActiveBlock(blk)
        setDone(prev => new Set([...prev, blk]))
      }
      else if (tick <= 3 + TOTAL_BLOCKS + 3) { setTransferPhase('d2h'); setActiveBlock(-1) }
      else { tick = 0; setDone(new Set()); setTransferPhase('idle') }
    }, 1000)
    return () => clearInterval(interval)
  }, [isInView])

  const activeRow = activeBlock >= 0 ? Math.floor(activeBlock / BLOCKS_X) : -1
  const activeCol = activeBlock >= 0 ? activeBlock % BLOCKS_X : -1

  return (
    <section ref={ref} className="min-h-screen flex items-center justify-center py-20 px-4">
      <div className="max-w-6xl w-full grid md:grid-cols-2 gap-12 items-center">
        <motion.div initial={{ opacity: 0, x: -40 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: false }}>
          <span className="text-xs uppercase tracking-widest text-emerald-400 font-semibold bg-emerald-500/10 px-3 py-1 rounded-full">Implementation 5 / 6</span>
          <h2 className="text-3xl md:text-4xl font-bold mt-4">
            <span className="text-emerald-400">CUDA</span> GPU Kernel
          </h2>
          <p className="text-gray-400 mt-4 leading-relaxed">
            Massively parallel — 8.29 million threads (one per pixel). Organized into 32,400 thread blocks. Each block loads a tile into fast shared memory before computing.
          </p>

          {/* Transfer pipeline */}
          <div className="mt-6 bg-gray-800/60 rounded-xl p-4 border border-gray-700/50">
            <p className="text-[10px] uppercase text-gray-500 mb-2">Data Pipeline</p>
            <div className="flex items-center gap-1">
              {[
                { id: 'h2d', label: 'CPU→GPU', color: '#a855f7' },
                { id: 'compute', label: 'Kernel', color: '#10b981' },
                { id: 'd2h', label: 'GPU→CPU', color: '#3b82f6' },
              ].map(s => (
                <div key={s.id} className={`flex-1 py-2 rounded text-center text-[10px] font-bold transition-all ${
                  transferPhase === s.id ? 'ring-2 scale-105' : 'opacity-40'
                }`} style={{ backgroundColor: s.color + '22', color: s.color, ringColor: s.color }}>
                  {s.label}
                </div>
              ))}
            </div>
          </div>

          {/* Memory hierarchy */}
          <div className="mt-4 grid grid-cols-3 gap-2">
            <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-2.5 text-center">
              <p className="text-[9px] text-gray-500 uppercase">Shared</p>
              <p className="text-[11px] text-emerald-300 font-bold">1.5 TB/s</p>
              <p className="text-[8px] text-gray-600">48KB/block</p>
            </div>
            <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-2.5 text-center">
              <p className="text-[9px] text-gray-500 uppercase">Constant</p>
              <p className="text-[11px] text-yellow-300 font-bold">Cached</p>
              <p className="text-[8px] text-gray-600">64KB total</p>
            </div>
            <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-2.5 text-center">
              <p className="text-[9px] text-gray-500 uppercase">Global</p>
              <p className="text-[11px] text-blue-300 font-bold">300 GB/s</p>
              <p className="text-[8px] text-gray-600">16GB GDDR6</p>
            </div>
          </div>

          <div className="mt-6 grid grid-cols-3 gap-3">
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Cores</p>
              <p className="text-lg font-bold text-emerald-400">2560</p>
            </div>
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Speedup</p>
              <p className="text-lg font-bold text-emerald-400">1555×</p>
            </div>
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Time</p>
              <p className="text-lg font-bold text-emerald-400">0.051s</p>
            </div>
          </div>
        </motion.div>

        {/* Right: Hierarchical Grid → Block → Threads */}
        <motion.div initial={{ opacity: 0, x: 40 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: false }}
          className="flex flex-col items-center gap-4">

          {/* Level 1: GRID */}
          <div className="w-full bg-gray-900/80 rounded-xl border border-gray-700/40 p-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs font-bold text-gray-300">GRID</p>
              <p className="text-[9px] text-gray-500 font-mono">gridDim(240, 135) — showing 8×4</p>
            </div>
            <div className="grid gap-1.5" style={{ gridTemplateColumns: `repeat(${BLOCKS_X}, 1fr)` }}>
              {Array.from({ length: TOTAL_BLOCKS }).map((_, i) => {
                const isActive = i === activeBlock
                const isDone = done.has(i)
                return (
                  <div key={i}
                    className={`w-10 h-10 md:w-12 md:h-12 rounded-md border-2 relative overflow-hidden transition-all duration-300 ${
                      isActive ? 'border-yellow-400 shadow-lg shadow-yellow-500/40 z-10 scale-110' :
                      isDone ? 'border-emerald-500/50' : 'border-gray-700/50'
                    }`}
                    style={{ backgroundColor: isActive ? '#fbbf2422' : isDone ? '#10b98122' : '#11182777' }}>
                    <div className="absolute inset-[3px] grid grid-cols-4 grid-rows-4 gap-[1px]">
                      {Array.from({ length: 16 }).map((_, t) => (
                        <div key={t} className={`rounded-[1px] transition-all duration-300 ${
                          isActive ? 'bg-yellow-300/80' : isDone ? 'bg-emerald-400/40' : 'bg-gray-700/20'
                        }`} />
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Arrow down */}
          {activeBlock >= 0 && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-gray-500 text-lg">▼</motion.div>
          )}

          {/* Level 2: BLOCK (zoomed) — shows threads as grid */}
          {activeBlock >= 0 && transferPhase === 'compute' && (
            <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
              className="w-full bg-gradient-to-br from-gray-900/90 to-yellow-950/20 rounded-xl border-2 border-yellow-500/40 p-5 shadow-lg shadow-yellow-500/10">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
                  <p className="text-sm font-bold text-yellow-300">BLOCK [{activeRow},{activeCol}]</p>
                </div>
                <p className="text-[10px] text-yellow-500/70 font-mono bg-yellow-500/10 px-2 py-0.5 rounded">blockDim(16,16) = 256 threads</p>
              </div>
              {/* 8x8 thread grid representing 16x16 (scaled down) */}
              <div className="grid grid-cols-8 gap-1 max-w-[320px] mx-auto">
                {Array.from({ length: 64 }).map((_, t) => {
                  const row = Math.floor(t / 8)
                  const col = t % 8
                  const wave = (row + col) * 0.04
                  return (
                    <motion.div key={t}
                      initial={{ opacity: 0, scale: 0, rotateY: 90 }}
                      animate={{ opacity: 1, scale: 1, rotateY: 0 }}
                      transition={{ delay: wave, type: 'spring', stiffness: 200 }}
                      className="w-8 h-8 md:w-9 md:h-9 rounded bg-gradient-to-br from-yellow-400/70 to-amber-500/50 border border-yellow-400/50 flex items-center justify-center shadow-sm shadow-yellow-500/20"
                      style={{ boxShadow: `0 0 6px ${t % 8 === col ? '#fbbf2433' : 'transparent'}` }}>
                      <span className="text-[7px] text-yellow-100 font-mono font-bold">{t}</span>
                    </motion.div>
                  )
                })}
              </div>
              <p className="text-[10px] text-yellow-400/60 text-center mt-3 font-medium">Each thread computes 1 output pixel</p>

              {/* Arrow down */}
              <motion.div animate={{ y: [0, 4, 0] }} transition={{ duration: 1.5, repeat: Infinity }}
                className="text-yellow-500/70 text-xl text-center mt-3 font-bold">▼</motion.div>

              {/* Level 3: Single THREAD — tile + halo view */}
              <div className="mt-3 border border-emerald-500/40 rounded-xl p-4 bg-gradient-to-br from-emerald-500/5 to-emerald-900/10 shadow-inner">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                    <p className="text-[11px] font-bold text-emerald-300">THREAD — Reads 3×3 from Shared Memory</p>
                  </div>
                  <p className="text-[9px] text-emerald-500/60 font-mono bg-emerald-500/10 px-2 py-0.5 rounded">tile + halo</p>
                </div>
                <div className="flex items-center gap-6 justify-center">
                  {/* 5x5 tile showing halo */}
                  <div className="grid grid-cols-5 gap-[3px]">
                    {Array.from({ length: 25 }).map((_, idx) => {
                      const r = Math.floor(idx / 5)
                      const c = idx % 5
                      const isHalo = r === 0 || r === 4 || c === 0 || c === 4
                      const isCenter = r === 2 && c === 2
                      const isKernel = r >= 1 && r <= 3 && c >= 1 && c <= 3
                      return (
                        <motion.div key={idx}
                          animate={{
                            scale: isCenter ? [1, 1.3, 1] : isKernel ? [1, 1.05, 1] : 1,
                            opacity: isHalo ? [0.5, 0.8, 0.5] : 1
                          }}
                          transition={{ duration: isCenter ? 0.8 : 2, repeat: Infinity, delay: isKernel ? (r + c) * 0.1 : 0 }}
                          className={`w-8 h-8 rounded border-2 flex items-center justify-center transition-all ${
                            isCenter ? 'bg-white/90 border-white ring-2 ring-white/60 shadow-lg shadow-white/30' :
                            isKernel ? 'bg-emerald-400/70 border-emerald-400/60 shadow-sm shadow-emerald-400/20' :
                            isHalo ? 'bg-red-400/40 border-red-500/40' :
                            'bg-gray-700/30 border-gray-600/20'
                          }`}>
                          {isCenter && <span className="text-[9px] text-gray-900 font-black">P</span>}
                          {isKernel && !isCenter && <span className="text-[6px] text-emerald-200/60">•</span>}
                        </motion.div>
                      )
                    })}
                  </div>
                  {/* Arrow and result */}
                  <div className="flex flex-col items-center gap-2">
                    <motion.span
                      animate={{ x: [0, 6, 0], opacity: [0.4, 1, 0.4] }}
                      transition={{ duration: 1.2, repeat: Infinity }}
                      className="text-emerald-400 text-lg font-bold">→</motion.span>
                    <motion.div
                      animate={{ scale: [1, 1.1, 1], rotate: [0, 5, -5, 0] }}
                      transition={{ duration: 2, repeat: Infinity }}
                      className="bg-yellow-500/20 border border-yellow-500/40 rounded px-2 py-1">
                      <span className="text-[9px] text-yellow-300 font-bold">×kernel</span>
                    </motion.div>
                    <motion.span
                      animate={{ x: [0, 6, 0], opacity: [0.4, 1, 0.4] }}
                      transition={{ duration: 1.2, repeat: Infinity, delay: 0.4 }}
                      className="text-emerald-400 text-lg font-bold">→</motion.span>
                  </div>
                  {/* Output pixel */}
                  <motion.div
                    animate={{ scale: [1, 1.15, 1], boxShadow: ['0 0 0px #10b981', '0 0 20px #10b981', '0 0 0px #10b981'] }}
                    transition={{ duration: 1.2, repeat: Infinity }}
                    className="w-14 h-14 rounded-lg border-2 border-emerald-400 bg-gradient-to-br from-emerald-500 to-emerald-600 flex flex-col items-center justify-center shadow-lg">
                    <span className="text-[10px] text-white font-bold">OUT</span>
                    <span className="text-[7px] text-emerald-200/70">pixel</span>
                  </motion.div>
                </div>
                {/* Legend */}
                <div className="flex justify-center gap-5 mt-4 pt-3 border-t border-gray-700/30">
                  <div className="flex items-center gap-1.5">
                    <div className="w-3.5 h-3.5 rounded bg-red-400/50 border border-red-500/40" />
                    <span className="text-[9px] text-gray-400">Halo</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-3.5 h-3.5 rounded bg-emerald-400/70 border border-emerald-400/60" />
                    <span className="text-[9px] text-gray-400">3×3 neighborhood</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-3.5 h-3.5 rounded bg-white/80 border border-white" />
                    <span className="text-[9px] text-gray-400">Output pixel</span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* PCIe indicator */}
          {(transferPhase === 'h2d' || transferPhase === 'd2h') && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="flex items-center gap-3 bg-purple-500/10 border border-purple-500/30 rounded-lg px-4 py-2">
              <motion.div animate={{ x: transferPhase === 'h2d' ? [0, 30, 0] : [30, 0, 30] }}
                transition={{ repeat: Infinity, duration: 1.5, ease: 'easeInOut' }}
                className="w-3 h-3 rounded-full bg-purple-400" />
              <span className="text-purple-300 text-xs">{transferPhase === 'h2d' ? 'Host → Device' : 'Device → Host'} (PCIe 3.0)</span>
            </motion.div>
          )}
        </motion.div>
      </div>
    </section>
  )
}
