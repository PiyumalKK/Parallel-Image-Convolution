import { motion, useInView } from 'framer-motion'
import { useRef, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Text } from '@react-three/drei'

function GPUGrid() {
  const blocks = []
  for (let x = 0; x < 8; x++) {
    for (let y = 0; y < 4; y++) {
      blocks.push(
        <mesh key={`${x}-${y}`} position={[x * 1.2 - 4.2, y * 1.2 - 1.8, 0]}>
          <boxGeometry args={[1, 1, 0.3]} />
          <meshStandardMaterial
            color={`hsl(${250 + x * 10 + y * 5}, 70%, ${55 + Math.random() * 15}%)`}
            transparent
            opacity={0.85}
          />
        </mesh>
      )
    }
  }
  return <group>{blocks}</group>
}

function ImagePlane() {
  return (
    <mesh position={[0, 0, -2]} rotation={[0, 0, 0]}>
      <planeGeometry args={[12, 7]} />
      <meshStandardMaterial color="#1a1a2e" transparent opacity={0.5} />
    </mesh>
  )
}

const implementations = [
  {
    id: 'openmp',
    name: 'OpenMP',
    color: '#22c55e',
    icon: '🔀',
    description: 'Shared-memory parallelism with compiler directives',
    details: [
      '#pragma omp parallel for collapse(2)',
      'Fork-join model — threads auto-created',
      'Dynamic scheduling for load balance',
      'No mutex needed (unique output indices)',
    ],
    speedup: '3.95×',
    time: '20.61s',
  },
  {
    id: 'posix',
    name: 'POSIX Pthreads',
    color: '#f97316',
    icon: '🧵',
    description: 'Explicit thread creation and manual row assignment',
    details: [
      'pthread_create() — manual thread lifecycle',
      'Static row-block partitioning',
      'ThreadArgs struct passes row ranges',
      'pthread_join() synchronizes at end',
    ],
    speedup: '4.04×',
    time: '20.16s',
  },
  {
    id: 'mpi',
    name: 'MPI',
    color: '#a855f7',
    icon: '📡',
    description: 'Distributed-memory with message passing',
    details: [
      'MPI_Bcast → broadcast kernel + dimensions',
      'MPI_Scatterv → distribute row chunks',
      'Each rank computes independently',
      'MPI_Gatherv → collect results at rank 0',
    ],
    speedup: '4.02×',
    time: '20.24s',
  },
  {
    id: 'cuda',
    name: 'CUDA',
    color: '#ef4444',
    icon: '🚀',
    description: 'GPU massive parallelism — 1 thread per pixel',
    details: [
      '16×16 thread blocks on Tesla T4 (40 SMs)',
      'Shared memory tiling with halo pixels',
      '__constant__ memory for kernel weights',
      '__syncthreads() for tile synchronization',
    ],
    speedup: '1054×',
    time: '0.077s',
  },
  {
    id: 'hybrid',
    name: 'Hybrid MPI+OpenMP',
    color: '#06b6d4',
    icon: '⚡',
    description: 'Two-level: MPI distributes, OpenMP parallelizes locally',
    details: [
      'Level 1: MPI scatters row chunks across ranks',
      'Level 2: OpenMP threads within each rank',
      'Best config: 4 ranks × 2 threads = 8 workers',
      'Optimized local convolution function',
    ],
    speedup: '18.1×',
    time: '4.49s',
  },
]

export default function ParallelFlow() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: false })
  const [activeImpl, setActiveImpl] = useState(0)

  const current = implementations[activeImpl]

  return (
    <section
      id="parallel-flow"
      ref={ref}
      className="min-h-screen flex items-center justify-center py-24 px-6"
    >
      <div className="max-w-7xl w-full">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: false }}
          className="text-4xl md:text-5xl font-bold text-center mb-16"
        >
          Five <span className="text-purple-400">Parallel</span> Implementations
        </motion.h2>

        {/* Implementation selector tabs */}
        <div className="flex flex-wrap justify-center gap-3 mb-12">
          {implementations.map((impl, i) => (
            <button
              key={impl.id}
              onClick={() => setActiveImpl(i)}
              className={`px-5 py-3 rounded-xl text-sm font-medium transition-all duration-300 ${
                activeImpl === i
                  ? 'scale-105 shadow-lg shadow-indigo-500/20'
                  : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
              }`}
              style={{
                backgroundColor: activeImpl === i ? `${impl.color}20` : undefined,
                borderColor: activeImpl === i ? impl.color : 'transparent',
                border: `1px solid ${activeImpl === i ? impl.color : 'transparent'}`,
                color: activeImpl === i ? impl.color : undefined,
              }}
            >
              <span className="mr-2">{impl.icon}</span>
              {impl.name}
            </button>
          ))}
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Left: Details */}
          <motion.div
            key={current.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4 }}
            className="bg-gray-900/60 rounded-2xl p-8 border border-gray-800"
          >
            <div className="flex items-center gap-3 mb-4">
              <span className="text-4xl">{current.icon}</span>
              <div>
                <h3 className="text-2xl font-bold" style={{ color: current.color }}>
                  {current.name}
                </h3>
                <p className="text-gray-400 text-sm">{current.description}</p>
              </div>
            </div>

            <div className="space-y-3 mt-6">
              {current.details.map((detail, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className="flex items-start gap-3"
                >
                  <span className="text-indigo-400 mt-1">▸</span>
                  <span className="text-gray-300 font-mono text-sm">{detail}</span>
                </motion.div>
              ))}
            </div>

            <div className="mt-8 grid grid-cols-2 gap-4">
              <div className="bg-gray-800/50 rounded-lg p-4 text-center">
                <p className="text-xs text-gray-500 uppercase tracking-wider">Speedup</p>
                <p className="text-3xl font-bold mt-1" style={{ color: current.color }}>
                  {current.speedup}
                </p>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4 text-center">
                <p className="text-xs text-gray-500 uppercase tracking-wider">Time (Blur)</p>
                <p className="text-3xl font-bold mt-1 text-gray-200">{current.time}</p>
              </div>
            </div>
          </motion.div>

          {/* Right: 3D Visualization */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: false }}
            className="bg-gray-900/60 rounded-2xl border border-gray-800 overflow-hidden"
            style={{ height: '500px' }}
          >
            {activeImpl === 3 ? (
              /* CUDA - show GPU grid */
              <Canvas camera={{ position: [0, 0, 8], fov: 50 }}>
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} />
                <GPUGrid />
                <ImagePlane />
                <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={1} />
              </Canvas>
            ) : (
              /* Other implementations - show row decomposition */
              <div className="h-full flex items-center justify-center p-8">
                <ThreadAnimation impl={current} />
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </section>
  )
}

function ThreadAnimation({ impl }) {
  const numWorkers = impl.id === 'hybrid' ? 8 : impl.id === 'cuda' ? 2560 : 4
  const displayWorkers = Math.min(numWorkers, 8)
  const colors = {
    openmp: ['#22c55e', '#16a34a', '#15803d', '#166534'],
    posix: ['#f97316', '#ea580c', '#c2410c', '#9a3412'],
    mpi: ['#a855f7', '#9333ea', '#7e22ce', '#6b21a8'],
    cuda: ['#ef4444'],
    hybrid: ['#06b6d4', '#0891b2', '#0e7490', '#155e75', '#22d3ee', '#67e8f9', '#a5f3fc', '#cffafe'],
  }

  const workerColors = colors[impl.id] || colors.openmp

  return (
    <div className="w-full">
      <div className="text-center mb-6">
        <p className="text-sm text-gray-400">
          Image split into <span className="font-bold text-white">{displayWorkers}</span> row blocks
        </p>
      </div>

      {/* Image representation with row blocks */}
      <div className="relative mx-auto" style={{ maxWidth: '300px' }}>
        <div className="border border-gray-600 rounded-lg overflow-hidden">
          {Array.from({ length: displayWorkers }).map((_, i) => (
            <motion.div
              key={i}
              initial={{ scaleX: 0 }}
              animate={{ scaleX: 1 }}
              transition={{ delay: i * 0.15, duration: 0.5 }}
              className="h-10 flex items-center justify-center text-xs font-mono text-white/80 border-b border-gray-700 last:border-b-0"
              style={{ backgroundColor: workerColors[i % workerColors.length] + '60' }}
            >
              {impl.id === 'hybrid'
                ? `Rank ${Math.floor(i / 2)} · Thread ${i % 2}`
                : impl.id === 'mpi'
                ? `Rank ${i}`
                : `Thread ${i}`}
            </motion.div>
          ))}
        </div>

        {/* Worker labels */}
        <div className="absolute -right-24 top-0 bottom-0 flex flex-col justify-around">
          {Array.from({ length: Math.min(displayWorkers, 4) }).map((_, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 + i * 0.1 }}
              className="text-xs text-gray-500"
            >
              ← Core {i}
            </motion.div>
          ))}
        </div>
      </div>

      {/* Arrows showing flow */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
        className="text-center mt-6 text-gray-500 text-sm"
      >
        {impl.id === 'mpi' && '↓ MPI_Scatterv → Compute → MPI_Gatherv ↓'}
        {impl.id === 'openmp' && '↓ Fork → #pragma parallel → Join ↓'}
        {impl.id === 'posix' && '↓ pthread_create → compute → pthread_join ↓'}
        {impl.id === 'hybrid' && '↓ Scatter → OMP parallel → Gather ↓'}
      </motion.div>
    </div>
  )
}
