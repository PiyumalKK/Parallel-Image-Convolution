import { motion } from 'framer-motion'

const conclusions = [
  {
    impl: 'OpenMP',
    color: '#22c55e',
    speedup: '4.05×',
    highlight: 'Best CPU speedup — near-perfect efficiency with minimal code changes',
    icon: '🔀',
  },
  {
    impl: 'POSIX Pthreads',
    color: '#f97316',
    speedup: '3.89×',
    highlight: 'Static partitioning — fine-grained control over thread lifecycle',
    icon: '🧵',
  },
  {
    impl: 'MPI',
    color: '#a855f7',
    speedup: '3.96×',
    highlight: 'Scalable across nodes but seam artefact (RMSE=1.40) needs halo-exchange fix',
    icon: '📡',
  },
  {
    impl: 'CUDA',
    color: '#ef4444',
    speedup: '1555×',
    highlight: 'Absolute best — 2560 GPU cores with shared-memory tiling',
    icon: '🚀',
  },
  {
    impl: 'Hybrid MPI+OMP',
    color: '#06b6d4',
    speedup: '18.3×',
    highlight: 'Super-linear efficiency (228%) — cache-friendly chunking on multi-node',
    icon: '⚡',
  },
]

export default function Conclusion() {
  return (
    <section
      id="conclusion"
      className="min-h-screen flex items-center justify-center py-24 px-6 relative"
    >
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-t from-indigo-950/20 to-transparent" />

      <div className="relative z-10 max-w-6xl w-full">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: false }}
          className="text-4xl md:text-5xl font-bold text-center mb-4"
        >
          <span className="text-emerald-400">Conclusion</span>
        </motion.h2>

        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: false }}
          transition={{ delay: 0.2 }}
          className="text-gray-400 text-center max-w-2xl mx-auto mb-16"
        >
          All implementations are perceptually correct (RMSE &lt; 1.5).
          The true bottleneck is hardware parallelism (4 cores), not Amdahl's sequential fraction.
        </motion.p>

        <div className="space-y-4">
          {conclusions.map((item, i) => (
            <motion.div
              key={item.impl}
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: false }}
              transition={{ delay: i * 0.1 }}
              className="flex items-center gap-6 bg-gray-900/60 rounded-xl p-6 border border-gray-800 hover:border-gray-600 transition-colors"
            >
              <span className="text-3xl">{item.icon}</span>
              <div className="flex-1">
                <h3 className="text-lg font-bold" style={{ color: item.color }}>
                  {item.impl}
                </h3>
                <p className="text-gray-400 text-sm">{item.highlight}</p>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold" style={{ color: item.color }}>
                  {item.speedup}
                </p>
                <p className="text-xs text-gray-500">speedup</p>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Final takeaway */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: false }}
          transition={{ delay: 0.6 }}
          className="mt-16 text-center"
        >
          <div className="inline-block bg-gradient-to-r from-indigo-500/10 to-purple-500/10 border border-indigo-500/30 rounded-2xl px-10 py-6">
            <p className="text-xl text-indigo-200 font-medium mb-2">
              Key Takeaway
            </p>
            <p className="text-gray-400 max-w-lg">
              Image convolution is embarrassingly parallel. With the right hardware (GPU)
              and proper memory optimization (shared memory tiling, constant memory),
              we achieve <span className="text-green-400 font-bold">three orders of magnitude</span> speedup
              over the serial baseline.
            </p>
          </div>
        </motion.div>

        {/* Footer */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: false }}
          transition={{ delay: 1 }}
          className="mt-16 text-center text-gray-600 text-sm"
        >
          <p>EE7218 / EC7207: High Performance Computing — University of Ruhuna</p>
          <p className="mt-1">Group 42 • 2026</p>
        </motion.div>
      </div>
    </section>
  )
}
