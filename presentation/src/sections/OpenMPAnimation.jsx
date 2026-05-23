import { motion, useInView } from 'framer-motion'
import { useRef, useState, useEffect } from 'react'

const GRID = 12
const NUM_THREADS = 4
const COLORS = ['#3b82f6', '#8b5cf6', '#06b6d4', '#f59e0b']
const ROWS_PER_THREAD = GRID / NUM_THREADS

export default function OpenMPAnimation() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: false, margin: '-100px' })
  const [threadPixels, setThreadPixels] = useState([0, 0, 0, 0])
  const [forkState, setForkState] = useState('single') // single → forked → joined

  useEffect(() => {
    if (!isInView) { setThreadPixels([0,0,0,0]); setForkState('single'); return }
    const t1 = setTimeout(() => setForkState('forked'), 800)
    const interval = setInterval(() => {
      setThreadPixels(prev => {
        const next = prev.map(p => {
          if (p >= ROWS_PER_THREAD * GRID) return ROWS_PER_THREAD * GRID
          return p + 1
        })
        if (next.every(p => p >= ROWS_PER_THREAD * GRID)) {
          setForkState('joined')
          clearInterval(interval)
          setTimeout(() => { setThreadPixels([0,0,0,0]); setForkState('single'); setTimeout(() => setForkState('forked'), 800) }, 2000)
        }
        return next
      })
    }, 150)
    return () => { clearTimeout(t1); clearInterval(interval) }
  }, [isInView])

  return (
    <section ref={ref} className="min-h-screen flex items-center justify-center py-20 px-4">
      <div className="max-w-6xl w-full grid md:grid-cols-2 gap-12 items-center">
        <motion.div initial={{ opacity: 0, x: -40 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: false }}>
          <span className="text-xs uppercase tracking-widest text-blue-400 font-semibold bg-blue-500/10 px-3 py-1 rounded-full">Implementation 2 / 6</span>
          <h2 className="text-3xl md:text-4xl font-bold mt-4">
            <span className="text-blue-400">OpenMP</span> Fork-Join
          </h2>
          <p className="text-gray-400 mt-4 leading-relaxed">
            A single pragma directive splits the image across 4 threads. Dynamic scheduling balances load — threads grab new rows as they finish.
          </p>

          {/* Fork-Join timeline */}
          <div className="mt-6 bg-gray-800/60 rounded-xl p-4 border border-gray-700/50">
            <p className="text-[10px] uppercase text-gray-500 mb-3">Fork-Join Model</p>
            <div className="relative h-16">
              {/* Main thread line */}
              <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-gray-600 -translate-y-1/2" />
              {/* Fork point */}
              <motion.div animate={{ scale: forkState === 'forked' ? [1, 1.3, 1] : 1 }}
                transition={{ duration: 0.5 }}
                className="absolute left-[15%] top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-blue-400 z-10" />
              {/* Join point */}
              <motion.div animate={{ scale: forkState === 'joined' ? [1, 1.3, 1] : 1 }}
                className="absolute right-[15%] top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-blue-400 z-10" />
              {/* Thread lines */}
              {forkState === 'forked' && COLORS.map((c, i) => (
                <motion.div key={i}
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  className="absolute h-0.5 origin-left"
                  style={{
                    left: '15%', right: '15%',
                    top: `${20 + i * 18}%`,
                    backgroundColor: c
                  }} />
              ))}
              <span className="absolute left-[15%] -bottom-1 text-[9px] text-blue-300 -translate-x-1/2">fork</span>
              <span className="absolute right-[15%] -bottom-1 text-[9px] text-blue-300 -translate-x-1/2">join</span>
            </div>
          </div>

          {/* Thread progress */}
          <div className="mt-5 space-y-1.5">
            {COLORS.map((c, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-[10px] font-mono text-gray-500 w-10">T-{i}</span>
                <div className="flex-1 h-3 bg-gray-800 rounded-full overflow-hidden">
                  <motion.div className="h-full rounded-full" style={{ backgroundColor: c, width: `${(threadPixels[i] / (ROWS_PER_THREAD * GRID)) * 100}%` }} />
                </div>
                <span className="text-[10px] text-gray-600 w-8 text-right">{Math.min(Math.round((threadPixels[i] / (ROWS_PER_THREAD * GRID)) * 100), 100)}%</span>
              </div>
            ))}
          </div>

          <div className="mt-6 grid grid-cols-3 gap-3">
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Threads</p>
              <p className="text-lg font-bold text-blue-400">4</p>
            </div>
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Speedup</p>
              <p className="text-lg font-bold text-blue-400">3.95×</p>
            </div>
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Time</p>
              <p className="text-lg font-bold text-blue-400">20.61s</p>
            </div>
          </div>
        </motion.div>

        {/* Grid */}
        <motion.div initial={{ opacity: 0, x: 40 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: false }}
          className="flex flex-col items-center">
          <div className="bg-gray-900/80 rounded-xl border border-gray-700/40 p-3">
            <div className="grid gap-[2px]" style={{ gridTemplateColumns: `repeat(${GRID}, 1fr)` }}>
              {Array.from({ length: GRID * GRID }).map((_, i) => {
                const row = Math.floor(i / GRID)
                const col = i % GRID
                const tid = Math.floor(row / ROWS_PER_THREAD)
                const localIdx = (row - tid * ROWS_PER_THREAD) * GRID + col
                const isDone = localIdx < threadPixels[tid]
                const isEdge = col === 0

                return (
                  <div key={i}
                    className="w-4 h-4 md:w-5 md:h-5 rounded-[2px] transition-all duration-300"
                    style={{
                      backgroundColor: isDone ? COLORS[tid] + 'BB' : COLORS[tid] + '15',
                      borderLeft: isEdge ? `2px solid ${COLORS[tid]}44` : 'none'
                    }}
                  />
                )
              })}
            </div>
            {/* Row labels */}
            <div className="flex justify-between mt-2 px-1">
              {COLORS.map((c, i) => (
                <div key={i} className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: c }} />
                  <span className="text-[9px]" style={{ color: c }}>T{i}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 bg-blue-500/10 border border-blue-500/30 rounded-lg px-4 py-2 text-center">
            <p className="text-blue-400 text-xs"><span className="font-bold">schedule(dynamic)</span> — threads grab work as available</p>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
