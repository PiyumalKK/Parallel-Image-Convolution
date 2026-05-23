import { motion, useInView } from 'framer-motion'
import { useRef, useState, useEffect } from 'react'

const GRID = 14
const K = 3

export default function SerialAnimation() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: false, margin: '-100px' })
  const [pixel, setPixel] = useState(0)
  const [sum, setSum] = useState(0)

  useEffect(() => {
    if (!isInView) { setPixel(0); setSum(0); return }
    const interval = setInterval(() => {
      setPixel(p => {
        const next = (p + 1) % (GRID * GRID)
        if (next === 0) setSum(0)
        else setSum(s => s + 1)
        return next
      })
    }, 200)
    return () => clearInterval(interval)
  }, [isInView])

  const px = pixel % GRID
  const py = Math.floor(pixel / GRID)

  return (
    <section ref={ref} className="min-h-screen flex items-center justify-center py-20 px-4">
      <div className="max-w-6xl w-full grid md:grid-cols-2 gap-12 items-center">
        <motion.div initial={{ opacity: 0, x: -40 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: false }}>
          <span className="text-xs uppercase tracking-widest text-orange-400 font-semibold bg-orange-500/10 px-3 py-1 rounded-full">Implementation 1 / 6</span>
          <h2 className="text-3xl md:text-4xl font-bold mt-4">
            <span className="text-orange-400">Serial</span> Convolution
          </h2>
          <p className="text-gray-400 mt-4 leading-relaxed">
            One thread, one pixel at a time. The kernel slides across the entire 3840×2160 image sequentially — no parallelism at all.
          </p>

          {/* Kernel visualization */}
          <div className="mt-6 bg-gray-800/60 rounded-xl p-4 border border-gray-700/50">
            <p className="text-[10px] uppercase text-gray-500 mb-2">21×21 Gaussian Kernel (shown 3×3)</p>
            <div className="grid grid-cols-3 gap-1 w-fit">
              {[0.075, 0.124, 0.075, 0.124, 0.204, 0.124, 0.075, 0.124, 0.075].map((v, i) => (
                <div key={i}
                  className="w-10 h-10 rounded flex items-center justify-center text-[10px] font-mono text-gray-300 transition-colors duration-300"
                  style={{ backgroundColor: pixel % 9 === i ? '#fb923c' : '#374151' }}>
                  {v.toFixed(3)}
                </div>
              ))}
            </div>
          </div>

          {/* Stats */}
          <div className="mt-6 grid grid-cols-3 gap-3">
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Pixels</p>
              <p className="text-lg font-bold text-orange-400">8.29M</p>
            </div>
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Ops/Pixel</p>
              <p className="text-lg font-bold text-orange-400">441</p>
            </div>
            <div className="bg-gray-800/40 rounded-lg p-3 text-center">
              <p className="text-[10px] text-gray-500 uppercase">Time</p>
              <p className="text-lg font-bold text-orange-400">81.43s</p>
            </div>
          </div>
        </motion.div>

        {/* Animated grid */}
        <motion.div initial={{ opacity: 0, x: 40 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: false }}
          className="flex flex-col items-center">
          <div className="relative p-2 bg-gray-900/80 rounded-xl border border-gray-700/40">
            <div className="grid gap-[2px]" style={{ gridTemplateColumns: `repeat(${GRID}, 1fr)` }}>
              {Array.from({ length: GRID * GRID }).map((_, i) => {
                const x = i % GRID
                const y = Math.floor(i / GRID)
                const isActive = i === pixel
                const isDone = i < pixel
                const inKernel = !isActive && Math.abs(x - px) <= 1 && Math.abs(y - py) <= 1

                return (
                  <div key={i}
                    className={`w-4 h-4 md:w-[18px] md:h-[18px] rounded-[2px] transition-all duration-300 ${
                      isActive ? 'bg-orange-400 shadow-md shadow-orange-500/60 scale-125' :
                      inKernel ? 'bg-yellow-500/50 ring-1 ring-yellow-400/50' :
                      isDone ? 'bg-orange-500/15' : 'bg-gray-700/40'
                    }`}
                  />
                )
              })}
            </div>

            {/* Scanline indicator */}
            <div
              className="absolute left-0 w-full h-[18px] border border-orange-400/30 rounded-sm pointer-events-none transition-all duration-300"
              style={{ top: `${8 + py * 20}px`, opacity: 0.5 }}
            />
          </div>

          {/* Progress bar */}
          <div className="w-full mt-4 space-y-1">
            <div className="flex justify-between text-[10px] text-gray-500">
              <span>Pixel {pixel + 1}</span>
              <span>{Math.round((pixel / (GRID * GRID)) * 100)}%</span>
            </div>
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
              <motion.div className="h-full bg-gradient-to-r from-orange-500 to-yellow-500 rounded-full"
                style={{ width: `${(pixel / (GRID * GRID)) * 100}%` }} />
            </div>
          </div>

          {/* Bottleneck callout */}
          <motion.div animate={{ opacity: [0.6, 1, 0.6] }} transition={{ repeat: Infinity, duration: 3, ease: 'easeInOut' }}
            className="mt-4 bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-2 text-center">
            <p className="text-red-400 text-xs font-semibold">⚠️ Only 1 core used out of 4 available</p>
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}
