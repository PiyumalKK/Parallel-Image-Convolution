import { motion, useInView } from 'framer-motion'
import { useRef, useState, useEffect } from 'react'

const GRID = 12
const NUM_THREADS = 4
const COLORS = ['#10b981', '#34d399', '#6ee7b7', '#a7f3d0']
const ROWS_PER = GRID / NUM_THREADS

export default function PthreadsAnimation() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: false, margin: '-100px' })
  const [phase, setPhase] = useState('idle') // idle → alloc → create → running → done → join → idle
  const [createdCount, setCreatedCount] = useState(0) // how many threads created so far
  const [joinedCount, setJoinedCount] = useState(0)
  const [progress, setProgress] = useState([0, 0, 0, 0])

  const intervalRef = useRef(null)
  const timeoutsRef = useRef([])
  const createdRef = useRef(0) // track created count for interval callback

  useEffect(() => {
    if (!isInView) {
      setPhase('idle'); setCreatedCount(0); setJoinedCount(0); setProgress([0,0,0,0])
      createdRef.current = 0
      if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
      timeoutsRef.current.forEach(t => clearTimeout(t))
      timeoutsRef.current = []
      return
    }

    function addTimeout(fn, ms) {
      const id = setTimeout(fn, ms)
      timeoutsRef.current.push(id)
      return id
    }

    function runCycle() {
      // Clear any leftover
      if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
      timeoutsRef.current.forEach(t => clearTimeout(t))
      timeoutsRef.current = []

      setPhase('idle'); setCreatedCount(0); setJoinedCount(0); setProgress([0,0,0,0])
      createdRef.current = 0

      // t=400: alloc phase
      addTimeout(() => setPhase('alloc'), 400)

      // t=1000: create phase — spawn threads one by one, each starts filling IMMEDIATELY
      addTimeout(() => {
        setPhase('create')

        // Start the fill interval — only fills threads that exist
        intervalRef.current = setInterval(() => {
          setProgress(prev => {
            const next = [...prev]
            let allDone = true
            for (let i = 0; i < NUM_THREADS; i++) {
              if (i < createdRef.current) {
                next[i] = Math.min(prev[i] + 1, ROWS_PER * GRID)
              }
              if (next[i] < ROWS_PER * GRID) allDone = false
            }
            // All threads done → stop and move to join
            if (allDone && createdRef.current === NUM_THREADS) {
              clearInterval(intervalRef.current)
              intervalRef.current = null
              setPhase('done')
              addTimeout(() => {
                setPhase('join')
                setJoinedCount(1)
                addTimeout(() => setJoinedCount(2), 300)
                addTimeout(() => setJoinedCount(3), 600)
                addTimeout(() => setJoinedCount(4), 900)
                addTimeout(() => runCycle(), 3500)
              }, 800)
            }
            return next
          })
        }, 200)

        // Stagger thread creation — each thread starts filling when created
        createdRef.current = 1; setCreatedCount(1)
        addTimeout(() => { createdRef.current = 2; setCreatedCount(2) }, 400)
        addTimeout(() => { createdRef.current = 3; setCreatedCount(3) }, 800)
        addTimeout(() => { createdRef.current = 4; setCreatedCount(4); setPhase('running') }, 1200)
      }, 1000)
    }

    runCycle()

    return () => {
      timeoutsRef.current.forEach(t => clearTimeout(t))
      timeoutsRef.current = []
      if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
    }
  }, [isInView])

  const getThreadState = (i) => {
    if (phase === 'idle') return 'idle'
    if (phase === 'alloc') return 'alloc'
    if (phase === 'create' || phase === 'running') {
      if (i >= createdCount) return 'pending'
      if (progress[i] >= ROWS_PER * GRID) return 'done'
      if (progress[i] > 0) return 'running'
      return 'created'
    }
    if (phase === 'done') return 'done'
    if (phase === 'join') return i < joinedCount ? 'joined' : 'done'
    return 'idle'
  }

  const stateColor = (s) => {
    switch(s) {
      case 'idle': return '#6b7280'
      case 'alloc': return '#f59e0b'
      case 'pending': return '#6b7280'
      case 'created': return '#fbbf24'
      case 'running': return '#10b981'
      case 'done': return '#06b6d4'
      case 'joined': return '#8b5cf6'
      default: return '#6b7280'
    }
  }

  const stateLabel = (s) => {
    switch(s) {
      case 'idle': return '—'
      case 'alloc': return 'ALLOC'
      case 'pending': return 'WAIT'
      case 'created': return 'SPAWNED'
      case 'running': return 'RUNNING'
      case 'done': return 'DONE'
      case 'joined': return 'JOINED'
      default: return '—'
    }
  }

  return (
    <section ref={ref} className="min-h-screen flex items-center justify-center py-20 px-4">
      <div className="max-w-6xl w-full grid md:grid-cols-2 gap-12 items-center">
        <motion.div initial={{ opacity: 0, x: -40 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: false }}>
          <span className="text-xs uppercase tracking-widest text-green-400 font-semibold bg-green-500/10 px-3 py-1 rounded-full">Implementation 3 / 6</span>
          <h2 className="text-3xl md:text-4xl font-bold mt-4">
            <span className="text-green-400">POSIX Threads</span>
          </h2>
          <p className="text-gray-400 mt-4 leading-relaxed">
            Manual thread management — explicit create, execute, and join. Each thread owns a fixed block of rows via a ThreadArgs struct passed at creation.
          </p>

          {/* Main thread timeline showing create → join */}
          <div className="mt-6 bg-gray-800/60 rounded-xl p-4 border border-gray-700/50">
            <p className="text-[10px] uppercase text-gray-500 mb-3">Thread Lifecycle (main thread perspective)</p>
            <div className="relative" style={{ height: '200px' }}>
              {/* Main thread line — segments accumulate color as phases progress */}
              {/* Segment 1: start → malloc (amber once alloc reached) */}
              <motion.div
                animate={{ backgroundColor: (phase !== 'idle') ? '#f59e0b' : '#374151' }}
                transition={{ duration: 0.3 }}
                className="absolute top-3 left-0 h-[2px]"
                style={{ width: '10%' }} />
              {/* Segment 2: malloc → create (green once create reached) */}
              <motion.div
                animate={{ backgroundColor: (phase === 'create' || phase === 'running' || phase === 'done' || phase === 'join') ? '#10b981' : '#374151' }}
                transition={{ duration: 0.3 }}
                className="absolute top-3 left-[10%] h-[2px]"
                style={{ width: '12%' }} />
              {/* Segment 3: create → join (cyan while running, purple once join) */}
              <motion.div
                animate={{ backgroundColor: (phase === 'join') ? '#8b5cf6' : (phase === 'running' || phase === 'done') ? '#6366f1' : '#374151' }}
                transition={{ duration: 0.3 }}
                className="absolute top-3 left-[22%] h-[2px]"
                style={{ right: '8%' }} />
              {/* Segment 4: join → end (purple once join) */}
              <motion.div
                animate={{ backgroundColor: phase === 'join' ? '#8b5cf6' : '#374151' }}
                transition={{ duration: 0.3 }}
                className="absolute top-3 right-0 h-[2px]"
                style={{ width: '8%' }} />
              <span className="absolute -top-1 left-0 text-[9px] font-mono text-gray-400">main()</span>

              {/* Phase markers on main line */}
              <motion.div
                animate={{ backgroundColor: phase === 'alloc' ? '#f59e0b' : '#4b5563', scale: phase === 'alloc' ? 1.4 : 1 }}
                className="absolute top-2 left-[10%] w-3 h-3 rounded-full z-10 border border-gray-600" />
              <span className="absolute top-7 left-[10%] text-[8px] text-amber-400 -translate-x-1/2">malloc</span>

              <motion.div
                animate={{ backgroundColor: (phase === 'create' || phase === 'running' || phase === 'done' || phase === 'join') ? '#10b981' : '#4b5563', scale: phase === 'create' ? 1.4 : 1 }}
                className="absolute top-2 left-[22%] w-3 h-3 rounded-full z-10 border border-gray-600" />
              <span className="absolute top-7 left-[22%] text-[8px] text-green-400 -translate-x-1/2">create()</span>

              <motion.div
                animate={{ backgroundColor: phase === 'join' ? '#8b5cf6' : '#4b5563', scale: phase === 'join' ? 1.4 : 1 }}
                className="absolute top-2 right-[8%] w-3 h-3 rounded-full z-10 border border-gray-600" />
              <span className="absolute top-7 right-[8%] text-[8px] text-purple-400 -translate-x-1/2">join()</span>

              {/* T0–T3: thread lines that grow from create to join = progress bars */}
              {COLORS.map((c, i) => {
                const ts = getThreadState(i)
                const pct = progress[i] / (ROWS_PER * GRID)
                const top = 50 + i * 30
                const isCreated = (phase === 'create' || phase === 'running' || phase === 'done' || phase === 'join') && i < createdCount

                return (
                  <div key={`thread-${i}`} className="absolute left-[28%] right-[14%]" style={{ top: `${top}px`, height: '14px' }}>
                    {/* Track background (gray) */}
                    <div className="absolute inset-x-0 h-[6px] rounded-full bg-gray-800 top-1/2 -translate-y-1/2" />
                    {/* Progress fill */}
                    {isCreated && (
                      <motion.div
                        initial={{ width: '0%' }}
                        animate={{ width: `${pct * 100}%` }}
                        transition={{ duration: 0.15, ease: 'linear' }}
                        className="absolute h-[6px] rounded-full top-1/2 -translate-y-1/2"
                        style={{ backgroundColor: c, opacity: phase === 'join' && i < joinedCount ? 0.5 : 1 }}
                      />
                    )}
                    {/* Thread label left */}
                    <span className="absolute -left-7 top-1/2 -translate-y-1/2 text-[9px] font-mono" style={{ color: c }}>T{i}</span>
                    {/* State badge right */}
                    <span className="absolute -right-12 top-1/2 -translate-y-1/2 text-[8px] font-bold w-11 text-right" style={{ color: stateColor(ts) }}>
                      {stateLabel(ts)}
                    </span>
                  </div>
                )
              })}

              {/* Active code annotation */}
              {phase === 'create' && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  className="absolute bottom-0 left-[15%] text-[8px] font-mono text-green-300 bg-green-500/10 px-1.5 py-0.5 rounded border border-green-500/20">
                  pthread_create(&tid[{createdCount > 0 ? createdCount - 1 : 0}], NULL, convolve, &args[{createdCount > 0 ? createdCount - 1 : 0}])
                </motion.div>
              )}
              {phase === 'join' && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  className="absolute bottom-0 right-[2%] text-[8px] font-mono text-purple-300 bg-purple-500/10 px-1.5 py-0.5 rounded border border-purple-500/20">
                  pthread_join(tid[{joinedCount > 0 ? joinedCount - 1 : 0}], NULL)
                </motion.div>
              )}
            </div>
          </div>

          <div className="mt-5 grid grid-cols-3 gap-3">
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Threads</p>
              <p className="text-lg font-bold text-green-400">4</p>
            </div>
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Speedup</p>
              <p className="text-lg font-bold text-green-400">3.86×</p>
            </div>
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Time</p>
              <p className="text-lg font-bold text-green-400">20.65s</p>
            </div>
          </div>
        </motion.div>

        {/* Right side: Grid + struct */}
        <motion.div initial={{ opacity: 0, x: 40 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: false }}
          className="flex flex-col items-center">
          <div className="bg-gray-900/80 rounded-xl border border-gray-700/40 p-3">
            {/* Thread color legend */}
            <div className="flex justify-center gap-3 mb-2">
              {COLORS.map((c, i) => (
                <div key={i} className="flex items-center gap-1">
                  <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: c }} />
                  <span className="text-[9px] text-gray-500">T{i}</span>
                </div>
              ))}
            </div>
            <div className="grid gap-[2px]" style={{ gridTemplateColumns: `repeat(${GRID}, 1fr)` }}>
              {Array.from({ length: GRID * GRID }).map((_, i) => {
                const row = Math.floor(i / GRID)
                const col = i % GRID
                const tid = Math.floor(row / ROWS_PER)
                const localIdx = (row - tid * ROWS_PER) * GRID + col
                const isDone = localIdx < progress[tid]

                return (
                  <div key={i}
                    className={`w-4 h-4 md:w-5 md:h-5 rounded-[2px] transition-all duration-200 ${row % ROWS_PER === 0 && row > 0 ? 'border-t border-dashed border-gray-600/60' : ''}`}
                    style={{ backgroundColor: isDone ? COLORS[tid] : '#1f2937', opacity: isDone ? 0.9 : 0.2 }}
                  />
                )
              })}
            </div>
          </div>

          {/* ThreadArgs struct visualization */}
          <div className="mt-4 bg-gray-800/60 rounded-lg p-3 border border-green-500/20 w-full max-w-[300px]">
            <p className="text-[9px] text-gray-500 uppercase mb-2">ThreadArgs struct (per thread)</p>
            <div className="font-mono text-[10px] space-y-1">
              <p className="text-gray-500">typedef struct {'{'}</p>
              <p className="text-green-300 pl-3">int start_row, end_row;</p>
              <p className="text-green-300 pl-3">unsigned char *input, *output;</p>
              <p className="text-green-300 pl-3">float *kernel;</p>
              <p className="text-green-300 pl-3">int width, height, channels;</p>
              <p className="text-gray-500">{'}'} ThreadArgs;</p>
            </div>
            {phase === 'running' && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                className="mt-2 border-t border-gray-700/50 pt-2 text-[9px] text-gray-400">
                <span className="text-green-400">args[0]</span>: rows 0–2 &nbsp;
                <span className="text-green-400">args[1]</span>: rows 3–5 &nbsp;
                <span className="text-green-400">args[2]</span>: rows 6–8 &nbsp;
                <span className="text-green-400">args[3]</span>: rows 9–11
              </motion.div>
            )}
          </div>
        </motion.div>
      </div>
    </section>
  )
}
