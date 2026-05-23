import { motion, useInView } from 'framer-motion'
import { useRef, useState, useEffect } from 'react'

const GRID = 12
const NUM_THREADS = 4
const COLORS = ['#10b981', '#34d399', '#6ee7b7', '#a7f3d0']
const ROWS_PER = GRID / NUM_THREADS

export default function PthreadsAnimation() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: false, margin: '-100px' })
  const [states, setStates] = useState(['init', 'init', 'init', 'init'])
  const [progress, setProgress] = useState([0, 0, 0, 0])

  useEffect(() => {
    if (!isInView) { setStates(['init','init','init','init']); setProgress([0,0,0,0]); return }
    // Lifecycle: init → create → running → done → join
    const timers = []
    for (let i = 0; i < NUM_THREADS; i++) {
      timers.push(setTimeout(() => setStates(s => { const n = [...s]; n[i] = 'create'; return n }), 500 + i * 300))
      timers.push(setTimeout(() => setStates(s => { const n = [...s]; n[i] = 'running'; return n }), 1000 + i * 300))
    }
    const interval = setInterval(() => {
      setProgress(prev => {
        const next = prev.map(p => Math.min(p + 1, ROWS_PER * GRID))
        if (next.every(p => p >= ROWS_PER * GRID)) {
          setStates(['done','done','done','done'])
          clearInterval(interval)
          setTimeout(() => {
            setStates(['join','join','join','join'])
            setTimeout(() => { setProgress([0,0,0,0]); setStates(['init','init','init','init']) }, 1500)
          }, 1000)
        }
        return next
      })
    }, 150)
    return () => { timers.forEach(clearTimeout); clearInterval(interval) }
  }, [isInView])

  const stateColor = (s) => s === 'init' ? '#6b7280' : s === 'create' ? '#fbbf24' : s === 'running' ? '#10b981' : s === 'done' ? '#06b6d4' : '#8b5cf6'
  const stateLabel = (s) => s === 'init' ? 'INIT' : s === 'create' ? 'CREATE' : s === 'running' ? 'RUN' : s === 'done' ? 'DONE' : 'JOIN'

  return (
    <section ref={ref} className="min-h-screen flex items-center justify-center py-20 px-4">
      <div className="max-w-6xl w-full grid md:grid-cols-2 gap-12 items-center">
        <motion.div initial={{ opacity: 0, x: -40 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: false }}>
          <span className="text-xs uppercase tracking-widest text-green-400 font-semibold bg-green-500/10 px-3 py-1 rounded-full">Implementation 3 / 6</span>
          <h2 className="text-3xl md:text-4xl font-bold mt-4">
            <span className="text-green-400">POSIX Threads</span>
          </h2>
          <p className="text-gray-400 mt-4 leading-relaxed">
            Manual thread management — explicit create, execute, and join. Each thread owns a fixed block of rows. Low-level control over the entire lifecycle.
          </p>

          {/* Thread state machine */}
          <div className="mt-6 bg-gray-800/60 rounded-xl p-4 border border-gray-700/50">
            <p className="text-[10px] uppercase text-gray-500 mb-3">Thread Lifecycle</p>
            <div className="space-y-2">
              {Array.from({ length: NUM_THREADS }).map((_, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-gray-500 w-14">tid[{i}]</span>
                  <motion.div
                    animate={{ backgroundColor: stateColor(states[i]) + '33', borderColor: stateColor(states[i]) }}
                    className="px-2 py-0.5 rounded border text-[10px] font-bold"
                    style={{ color: stateColor(states[i]) }}>
                    {stateLabel(states[i])}
                  </motion.div>
                  <div className="flex-1 h-2.5 bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full rounded-full transition-all duration-300" style={{ width: `${(progress[i] / (ROWS_PER * GRID)) * 100}%`, backgroundColor: COLORS[i] }} />
                  </div>
                  <span className="text-[10px] text-gray-600 w-12 text-right">rows {i*ROWS_PER}–{(i+1)*ROWS_PER-1}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-6 grid grid-cols-3 gap-3">
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Threads</p>
              <p className="text-lg font-bold text-green-400">4</p>
            </div>
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Speedup</p>
              <p className="text-lg font-bold text-green-400">4.04×</p>
            </div>
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Time</p>
              <p className="text-lg font-bold text-green-400">20.16s</p>
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
                const tid = Math.floor(row / ROWS_PER)
                const localIdx = (row - tid * ROWS_PER) * GRID + col
                const isDone = localIdx < progress[tid]
                const isBorder = row % ROWS_PER === 0 && col === 0

                return (
                  <div key={i}
                    className={`w-4 h-4 md:w-5 md:h-5 rounded-[2px] transition-all duration-300 ${row % ROWS_PER === 0 ? 'border-t border-gray-600/50' : ''}`}
                    style={{ backgroundColor: isDone ? COLORS[tid] : '#1f2937', opacity: isDone ? 0.9 : 0.25 }}
                  />
                )
              })}
            </div>
          </div>

          {/* Struct visualization */}
          <div className="mt-4 bg-gray-800/60 rounded-lg p-3 border border-green-500/20 w-full max-w-[280px]">
            <p className="text-[9px] text-gray-500 uppercase mb-1">Thread Arg Struct</p>
            <div className="font-mono text-[10px] text-green-300 space-y-0.5">
              <p>{"{"} start_row, end_row,</p>
              <p>  input, output, kernel {"}"}</p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
