import { motion, useInView } from 'framer-motion'
import { useRef, useState } from 'react'

const kernels = [
  {
    name: 'Gaussian Blur',
    desc: 'Smoothing — reduces noise by averaging neighbors',
    color: 'blue',
    size: '21×21',
    ops: '~3.67B FLOPs',
    effect: 'Soft, smooth image',
    matrix: [
      [1, 2, 1],
      [2, 4, 2],
      [1, 2, 1],
    ],
  },
  {
    name: 'Edge Detection',
    desc: 'Sobel operator — highlights intensity gradients',
    color: 'red',
    size: '21×21',
    ops: '~3.67B FLOPs',
    effect: 'Edges/outlines only',
    matrix: [
      [-1, -1, -1],
      [-1, 8, -1],
      [-1, -1, -1],
    ],
  },
  {
    name: 'Sharpen',
    desc: 'Enhances edges by amplifying differences',
    color: 'green',
    size: '21×21',
    ops: '~3.67B FLOPs',
    effect: 'Crisper details',
    matrix: [
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0],
    ],
  },
]

export default function DemoSection() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: false })
  const [active, setActive] = useState(0)
  const kernel = kernels[active]

  const colorMap = {
    blue: { bg: 'bg-blue-500/10', border: 'border-blue-500/30', text: 'text-blue-400', glow: 'shadow-blue-500/20' },
    red: { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400', glow: 'shadow-red-500/20' },
    green: { bg: 'bg-green-500/10', border: 'border-green-500/30', text: 'text-green-400', glow: 'shadow-green-500/20' },
  }
  const colors = colorMap[kernel.color]

  return (
    <section ref={ref} id="demo" className="min-h-screen flex items-center justify-center py-20 px-6">
      <div className="max-w-5xl w-full">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: false }}
          className="text-center mb-12"
        >
          <span className="text-sm uppercase tracking-widest text-indigo-400 font-semibold">Project Demo</span>
          <h2 className="text-4xl md:text-5xl font-bold mt-2">
            Image <span className="text-indigo-400">Convolution</span> Results
          </h2>
          <p className="text-gray-400 mt-4 max-w-xl mx-auto">
            3840×2160 (4K UHD) test image — 8.3 million pixels × 3 channels × 21×21 kernel
          </p>
        </motion.div>

        {/* Kernel selector */}
        <div className="flex justify-center gap-4 mb-10">
          {kernels.map((k, i) => (
            <button
              key={k.name}
              onClick={() => setActive(i)}
              className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-all duration-300 ${
                i === active
                  ? `${colorMap[k.color].bg} ${colorMap[k.color].border} ${colorMap[k.color].text} border shadow-lg ${colorMap[k.color].glow}`
                  : 'bg-gray-800/50 text-gray-400 border border-gray-700 hover:border-gray-600'
              }`}
            >
              {k.name}
            </button>
          ))}
        </div>

        <div className="grid md:grid-cols-2 gap-8 items-center">
          {/* Kernel matrix visualization */}
          <motion.div
            key={active}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
            className={`${colors.bg} ${colors.border} border rounded-2xl p-8`}
          >
            <h3 className={`${colors.text} font-semibold text-lg mb-2`}>{kernel.name}</h3>
            <p className="text-gray-400 text-sm mb-6">{kernel.desc}</p>

            {/* 3x3 kernel preview */}
            <div className="flex justify-center mb-6">
              <div className="grid grid-cols-3 gap-1">
                {kernel.matrix.flat().map((val, i) => (
                  <div
                    key={i}
                    className={`w-12 h-12 flex items-center justify-center rounded-md font-mono text-sm font-bold ${
                      val > 0 ? `${colors.bg} ${colors.text}` :
                      val < 0 ? 'bg-red-500/10 text-red-400' :
                      'bg-gray-800 text-gray-600'
                    }`}
                  >
                    {val}
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-500">Kernel Size</span>
                <span className="text-gray-300 font-mono">{kernel.size}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Operations</span>
                <span className="text-gray-300 font-mono">{kernel.ops}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Effect</span>
                <span className={colors.text}>{kernel.effect}</span>
              </div>
            </div>
          </motion.div>

          {/* Before/After visualization */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: false }}
            className="space-y-6"
          >
            {/* Simulated before/after */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
                <p className="text-xs text-gray-500 uppercase mb-3">Input (3840×2160)</p>
                <div className="aspect-video bg-gradient-to-br from-gray-600 to-gray-800 rounded-lg relative overflow-hidden">
                  <div className="absolute inset-0 grid grid-cols-8 grid-rows-5 gap-px opacity-30">
                    {Array.from({ length: 40 }).map((_, i) => (
                      <div key={i} className="bg-gray-500" style={{ opacity: 0.3 + Math.random() * 0.7 }} />
                    ))}
                  </div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-gray-400 text-xs">Original</span>
                  </div>
                </div>
              </div>
              <div className={`rounded-xl p-4 border ${colors.bg} ${colors.border}`}>
                <p className={`text-xs uppercase mb-3 ${colors.text}`}>Output ({kernel.name})</p>
                <div className={`aspect-video rounded-lg relative overflow-hidden ${
                  kernel.color === 'blue' ? 'bg-gradient-to-br from-blue-900/50 to-blue-950/80' :
                  kernel.color === 'red' ? 'bg-gradient-to-br from-gray-900 to-black' :
                  'bg-gradient-to-br from-gray-700 to-gray-900'
                }`}>
                  <div className="absolute inset-0 grid grid-cols-8 grid-rows-5 gap-px opacity-30">
                    {Array.from({ length: 40 }).map((_, i) => (
                      <div key={i} style={{
                        backgroundColor: kernel.color === 'blue' ? '#3b82f6' :
                                        kernel.color === 'red' ? (Math.random() > 0.7 ? '#fff' : '#000') :
                                        '#9ca3af',
                        opacity: kernel.color === 'red' ? (Math.random() > 0.6 ? 0.8 : 0.05) :
                                 0.3 + Math.random() * 0.4
                      }} />
                    ))}
                  </div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className={`text-xs ${colors.text}`}>{kernel.effect}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Timing comparison */}
            <div className="bg-gray-900/60 rounded-xl p-5 border border-gray-800">
              <p className="text-xs text-gray-500 uppercase mb-3">Execution Time Comparison</p>
              <div className="space-y-3">
                {[
                  { name: 'Serial', time: '79.78s', width: '100%', color: '#6b7280' },
                  { name: 'OpenMP', time: '19.96s', width: '25.0%', color: '#22c55e' },
                  { name: 'CUDA', time: '0.051s', width: '0.1%', color: '#10b981' },
                ].map((item) => (
                  <div key={item.name} className="flex items-center gap-3">
                    <span className="text-xs text-gray-400 w-14">{item.name}</span>
                    <div className="flex-1 h-3 bg-gray-800 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        whileInView={{ width: item.width }}
                        viewport={{ once: false }}
                        transition={{ duration: 1, delay: 0.3 }}
                        className="h-full rounded-full"
                        style={{ backgroundColor: item.color, minWidth: item.name === 'CUDA' ? '4px' : undefined }}
                      />
                    </div>
                    <span className="text-xs font-mono text-gray-300 w-16 text-right">{item.time}</span>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  )
}
