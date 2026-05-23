import { motion, useInView } from 'framer-motion'
import { useRef, useState, useEffect } from 'react'

const KERNEL = [
  [1, 2, 1],
  [2, 4, 2],
  [1, 2, 1],
]

const IMAGE_GRID = [
  [120, 130, 125, 140, 135, 128, 132],
  [110, 145, 150, 155, 148, 142, 138],
  [105, 140, 200, 210, 195, 150, 130],
  [100, 135, 190, 220, 205, 145, 125],
  [108, 130, 180, 200, 185, 140, 128],
  [115, 125, 140, 150, 145, 135, 130],
  [120, 122, 128, 132, 130, 128, 125],
]

export default function ConvolutionDemo() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: false, margin: '-100px' })
  const [activePixel, setActivePixel] = useState({ x: 3, y: 3 })
  const [step, setStep] = useState(0)

  useEffect(() => {
    if (!isInView) return
    const interval = setInterval(() => {
      setStep((s) => {
        if (s >= 8) {
          // Move to next pixel
          setActivePixel((p) => {
            const nx = p.x + 1 > 4 ? 2 : p.x + 1
            const ny = nx === 2 ? (p.y + 1 > 4 ? 2 : p.y + 1) : p.y
            return { x: nx, y: ny }
          })
          return 0
        }
        return s + 1
      })
    }, 400)
    return () => clearInterval(interval)
  }, [isInView])

  const computeOutput = (cx, cy) => {
    let sum = 0
    let wSum = 0
    for (let ky = -1; ky <= 1; ky++) {
      for (let kx = -1; kx <= 1; kx++) {
        const px = cx + kx
        const py = cy + ky
        if (px >= 0 && px < 7 && py >= 0 && py < 7) {
          sum += IMAGE_GRID[py][px] * KERNEL[ky + 1][kx + 1]
          wSum += KERNEL[ky + 1][kx + 1]
        }
      }
    }
    return Math.round(sum / wSum)
  }

  return (
    <section
      id="convolution"
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
          What is <span className="text-indigo-400">Convolution</span>?
        </motion.h2>

        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: false }}
          transition={{ delay: 0.2 }}
          className="text-gray-400 text-center max-w-2xl mx-auto mb-16"
        >
          A kernel (filter) slides over each pixel, multiplying overlapping values
          and summing them to produce a new pixel. This operation is{' '}
          <span className="text-indigo-300 font-semibold">embarrassingly parallel</span> —
          every output pixel is independent.
        </motion.p>

        <div className="grid md:grid-cols-3 gap-8 items-center">
          {/* Input Image Grid */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: false }}
            className="flex flex-col items-center"
          >
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
              Input Image
            </h3>
            <div className="grid grid-cols-7 gap-0.5">
              {IMAGE_GRID.map((row, y) =>
                row.map((val, x) => {
                  const isActive =
                    Math.abs(x - activePixel.x) <= 1 && Math.abs(y - activePixel.y) <= 1
                  const isCenter = x === activePixel.x && y === activePixel.y
                  return (
                    <div
                      key={`${x}-${y}`}
                      className={`w-10 h-10 flex items-center justify-center text-xs font-mono rounded transition-all duration-300 ${
                        isCenter
                          ? 'bg-indigo-500 text-white scale-110 ring-2 ring-indigo-300'
                          : isActive
                          ? 'bg-indigo-500/30 text-indigo-200 ring-1 ring-indigo-500/50'
                          : 'bg-gray-800 text-gray-500'
                      }`}
                    >
                      {val}
                    </div>
                  )
                })
              )}
            </div>
          </motion.div>

          {/* Kernel + Operation */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: false }}
            transition={{ delay: 0.3 }}
            className="flex flex-col items-center"
          >
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
              Kernel (3×3 Gaussian)
            </h3>
            <div className="grid grid-cols-3 gap-1 mb-6">
              {KERNEL.map((row, y) =>
                row.map((val, x) => (
                  <div
                    key={`k-${x}-${y}`}
                    className="w-12 h-12 flex items-center justify-center text-sm font-bold rounded bg-purple-600/30 text-purple-200 border border-purple-500/40"
                  >
                    {val}
                  </div>
                ))
              )}
            </div>

            <div className="text-center">
              <div className="text-2xl mb-2">⊗</div>
              <div className="bg-gray-800/60 rounded-lg p-4 border border-gray-700">
                <p className="text-xs text-gray-400 mb-2">Output pixel:</p>
                <motion.div
                  key={`${activePixel.x}-${activePixel.y}`}
                  initial={{ scale: 0.5, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className="text-3xl font-bold text-green-400"
                >
                  {computeOutput(activePixel.x, activePixel.y)}
                </motion.div>
              </div>
            </div>
          </motion.div>

          {/* Formula */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: false }}
            transition={{ delay: 0.4 }}
            className="flex flex-col items-center"
          >
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
              The Formula
            </h3>
            <div className="bg-gray-900/80 rounded-xl p-6 border border-gray-700 font-mono text-sm">
              <p className="text-indigo-300 mb-2">O(x,y) = Σ Σ I(x+kx, y+ky) × K(kx,ky)</p>
              <hr className="border-gray-700 my-3" />
              <p className="text-gray-400 text-xs">For 3840×2160 image:</p>
              <p className="text-yellow-300 text-xs mt-1">= 8.3M pixels × 441 ops</p>
              <p className="text-red-300 text-xs">= 3.5 Billion operations</p>
              <hr className="border-gray-700 my-3" />
              <p className="text-gray-400 text-xs">Serial: <span className="text-red-400">81.43 seconds</span></p>
              <p className="text-gray-400 text-xs">CUDA: <span className="text-green-400">0.077 seconds</span></p>
              <p className="text-green-300 text-xs font-bold mt-1">→ 1054× faster!</p>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  )
}
