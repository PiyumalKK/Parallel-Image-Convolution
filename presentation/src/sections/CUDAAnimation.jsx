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
  const [transferPhase, setTransferPhase] = useState('idle') // idle → h2d → compute → d2h → idle

  useEffect(() => {
    if (!isInView) { setActiveBlock(-1); setDone(new Set()); setTransferPhase('idle'); return }
    let tick = 0
    const interval = setInterval(() => {
      tick++
      if (tick <= 8) { setTransferPhase('h2d') }
      else if (tick <= 8 + TOTAL_BLOCKS) {
        setTransferPhase('compute')
        const blk = tick - 9
        setActiveBlock(blk)
        setDone(prev => new Set([...prev, blk]))
      }
      else if (tick <= 8 + TOTAL_BLOCKS + 8) { setTransferPhase('d2h'); setActiveBlock(-1) }
      else { tick = 0; setDone(new Set()); setTransferPhase('idle') }
    }, 150)
    return () => clearInterval(interval)
  }, [isInView])

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

        {/* Block grid */}
        <motion.div initial={{ opacity: 0, x: 40 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: false }}
          className="flex flex-col items-center">
          <p className="text-[10px] text-gray-500 mb-2">Thread Block Grid (8×4 of 240×135 total)</p>
          <div className="bg-gray-900/80 rounded-xl border border-gray-700/40 p-3">
            <div className="grid gap-1.5" style={{ gridTemplateColumns: `repeat(${BLOCKS_X}, 1fr)` }}>
              {Array.from({ length: TOTAL_BLOCKS }).map((_, i) => {
                const isActive = i === activeBlock
                const isDone = done.has(i)
                return (
                  <div key={i}
                    className={`w-8 h-8 md:w-10 md:h-10 rounded-md border relative overflow-hidden transition-all duration-300 ${
                      isActive ? 'border-yellow-400 shadow-lg shadow-yellow-500/40 z-10 scale-110' :
                      isDone ? 'border-emerald-500/50' : 'border-gray-700/50'
                    }`}
                    style={{ backgroundColor: isActive ? '#fbbf2422' : isDone ? '#10b98122' : '#11182777' }}>
                    {/* Mini thread grid */}
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
          <p className="text-[10px] text-gray-500 mt-2">Each block = 16×16 = 256 threads</p>

          {/* PCIe indicator */}
          {(transferPhase === 'h2d' || transferPhase === 'd2h') && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="mt-4 flex items-center gap-3 bg-purple-500/10 border border-purple-500/30 rounded-lg px-4 py-2">
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
