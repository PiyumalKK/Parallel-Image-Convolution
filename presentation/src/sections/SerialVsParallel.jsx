import { motion, useInView } from 'framer-motion'
import { useRef, useState, useEffect } from 'react'

export default function SerialVsParallel() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: false, margin: '-100px' })
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    if (!isInView) {
      setProgress(0)
      return
    }
    const interval = setInterval(() => {
      setProgress((p) => (p >= 100 ? 0 : p + 2))
    }, 80)
    return () => clearInterval(interval)
  }, [isInView])

  const serialProgress = progress
  const parallelProgress = Math.min(progress * 4, 100)

  return (
    <section
      id="serial-vs-parallel"
      ref={ref}
      className="min-h-screen flex items-center justify-center py-24 px-6"
    >
      <div className="max-w-6xl w-full">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: false }}
          className="text-4xl md:text-5xl font-bold text-center mb-4"
        >
          Serial vs <span className="text-green-400">Parallel</span>
        </motion.h2>

        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: false }}
          transition={{ delay: 0.2 }}
          className="text-gray-400 text-center max-w-2xl mx-auto mb-16"
        >
          The serial implementation processes pixels one by one.
          Parallel splits the work across multiple cores simultaneously.
        </motion.p>

        <div className="grid md:grid-cols-2 gap-12">
          {/* Serial */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: false }}
            className="bg-gray-900/50 rounded-2xl p-8 border border-gray-800"
          >
            <h3 className="text-2xl font-bold text-red-400 mb-6">Serial (1 core)</h3>
            <div className="relative mb-6">
              {/* Image grid representation */}
              <div className="grid grid-cols-16 gap-0.5">
                {Array.from({ length: 64 }).map((_, i) => (
                  <div
                    key={i}
                    className={`h-3 rounded-sm transition-all duration-100 ${
                      i < Math.floor(serialProgress * 0.64)
                        ? 'bg-red-500'
                        : 'bg-gray-700'
                    }`}
                    style={{ width: '100%' }}
                  />
                ))}
              </div>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-400">Progress</span>
              <span className="text-red-400 font-mono">{serialProgress}%</span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-3 mt-2">
              <div
                className="bg-gradient-to-r from-red-600 to-red-400 h-3 rounded-full transition-all duration-100"
                style={{ width: `${serialProgress}%` }}
              />
            </div>
            <p className="text-gray-500 text-sm mt-4 font-mono">
              1 thread → processes row by row sequentially
            </p>
            <p className="text-red-300 text-lg font-bold mt-2">79.78 seconds</p>
          </motion.div>

          {/* Parallel */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: false }}
            className="bg-gray-900/50 rounded-2xl p-8 border border-indigo-500/30"
          >
            <h3 className="text-2xl font-bold text-green-400 mb-6">Parallel (4 cores)</h3>
            <div className="space-y-2 mb-6">
              {[0, 1, 2, 3].map((core) => (
                <div key={core} className="flex items-center gap-2">
                  <span className="text-xs text-gray-500 w-16">Core {core}</span>
                  <div className="flex-1 grid grid-cols-16 gap-0.5">
                    {Array.from({ length: 16 }).map((_, i) => (
                      <div
                        key={i}
                        className={`h-3 rounded-sm transition-all duration-100 ${
                          i < Math.floor(parallelProgress * 0.16)
                            ? `bg-${['green', 'blue', 'purple', 'cyan'][core]}-500`
                            : 'bg-gray-700'
                        }`}
                        style={{
                          backgroundColor:
                            i < Math.floor(parallelProgress * 0.16)
                              ? ['#22c55e', '#3b82f6', '#a855f7', '#06b6d4'][core]
                              : undefined,
                        }}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-400">Progress</span>
              <span className="text-green-400 font-mono">{parallelProgress}%</span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-3 mt-2">
              <div
                className="bg-gradient-to-r from-green-600 to-green-400 h-3 rounded-full transition-all duration-100"
                style={{ width: `${parallelProgress}%` }}
              />
            </div>
            <p className="text-gray-500 text-sm mt-4 font-mono">
              4 threads → each processes ¼ of rows simultaneously
            </p>
            <p className="text-green-300 text-lg font-bold mt-2">~19.96 seconds (4× faster)</p>
          </motion.div>
        </div>

        {/* Key insight */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: false }}
          transition={{ delay: 0.5 }}
          className="mt-12 text-center"
        >
          <div className="inline-block bg-indigo-500/10 border border-indigo-500/30 rounded-xl px-8 py-4">
            <p className="text-indigo-300 font-semibold">
              Each pixel computation is independent → No data dependencies → Perfect parallelism
            </p>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
