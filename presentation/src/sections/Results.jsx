import { motion, useInView } from 'framer-motion'
import { useRef, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, Cell
} from 'recharts'

const timeData = [
  { workers: 1, OpenMP: 78.68, POSIX: 78.44, MPI: 80.81 },
  { workers: 2, OpenMP: 39.57, POSIX: 39.80, MPI: 40.39 },
  { workers: 4, OpenMP: 19.96, POSIX: 20.65, MPI: 20.86 },
  { workers: 8, OpenMP: 19.72, POSIX: 20.53, MPI: 20.16 },
]

const speedupData = [
  { workers: 1, OpenMP: 1.01, POSIX: 1.02, MPI: 0.99, Ideal: 1 },
  { workers: 2, OpenMP: 2.02, POSIX: 2.01, MPI: 1.98, Ideal: 2 },
  { workers: 4, OpenMP: 4.00, POSIX: 3.86, MPI: 3.82, Ideal: 4 },
  { workers: 8, OpenMP: 4.05, POSIX: 3.88, MPI: 3.96, Ideal: 8 },
]

const cudaComparison = [
  { filter: 'Gaussian Blur', Serial: 79.78, CUDA: 0.0513, speedup: '1555×' },
  { filter: 'Edge Detection', Serial: 2.17, CUDA: 0.0120, speedup: '181×' },
  { filter: 'Sharpen', Serial: 0.26, CUDA: 0.0045, speedup: '57×' },
]

const efficiencyData = [
  { impl: 'OpenMP\n(4 workers)', efficiency: 99.9 },
  { impl: 'POSIX\n(4 workers)', efficiency: 96.6 },
  { impl: 'MPI\n(4 workers)', efficiency: 95.6 },
  { impl: 'CUDA\n(2560 cores)', efficiency: 60.7 },
  { impl: 'Hybrid\n(8 workers)', efficiency: 228 },
]

const rmseData = [
  { impl: 'OpenMP', blur: 0, edge: 0, sharpen: 0 },
  { impl: 'POSIX', blur: 0, edge: 0, sharpen: 0 },
  { impl: 'MPI', blur: 0.245, edge: 0.382, sharpen: 1.404 },
  { impl: 'CUDA', blur: 0.002, edge: 0, sharpen: 0 },
  { impl: 'Hybrid', blur: 0.115, edge: 0.202, sharpen: 0.790 },
]

export default function Results() {
  const ref = useRef(null)
  const [activeChart, setActiveChart] = useState(0)

  const charts = [
    { label: 'Execution Time', id: 'time' },
    { label: 'Speedup', id: 'speedup' },
    { label: 'CUDA vs Serial', id: 'cuda' },
    { label: 'Efficiency', id: 'efficiency' },
    { label: 'Accuracy (RMSE)', id: 'rmse' },
  ]

  return (
    <section
      id="results"
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
          Performance <span className="text-yellow-400">Results</span>
        </motion.h2>

        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: false }}
          transition={{ delay: 0.2 }}
          className="text-gray-400 text-center max-w-2xl mx-auto mb-12"
        >
          Benchmarked on Azure VM (4 cores) + NVIDIA Tesla T4 GPU
        </motion.p>

        {/* Chart tabs */}
        <div className="flex flex-wrap justify-center gap-2 mb-8">
          {charts.map((chart, i) => (
            <button
              key={chart.id}
              onClick={() => setActiveChart(i)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeChart === i
                  ? 'bg-indigo-500 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {chart.label}
            </button>
          ))}
        </div>

        {/* Chart display */}
        <motion.div
          key={activeChart}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-900/60 rounded-2xl p-8 border border-gray-800"
          style={{ height: '450px' }}
        >
          {activeChart === 0 && <TimeChart />}
          {activeChart === 1 && <SpeedupChart />}
          {activeChart === 2 && <CudaChart />}
          {activeChart === 3 && <EfficiencyChart />}
          {activeChart === 4 && <RMSEChart />}
        </motion.div>

        {/* Key metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
          {[
            { label: 'Best CPU Speedup', value: '4.04×', sub: 'POSIX @ 4 threads', color: 'text-orange-400' },
            { label: 'CUDA Speedup', value: '1054×', sub: 'Tesla T4 GPU', color: 'text-red-400' },
            { label: 'Hybrid Speedup', value: '18.1×', sub: '4P × 2T config', color: 'text-cyan-400' },
            { label: 'Max RMSE', value: '1.40', sub: 'MPI sharpen (seam)', color: 'text-purple-400' },
          ].map((metric) => (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: false }}
              className="bg-gray-800/50 rounded-xl p-4 text-center border border-gray-700"
            >
              <p className="text-xs text-gray-500 uppercase">{metric.label}</p>
              <p className={`text-2xl font-bold mt-1 ${metric.color}`}>{metric.value}</p>
              <p className="text-xs text-gray-500 mt-1">{metric.sub}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

function TimeChart() {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={timeData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
        <XAxis dataKey="workers" stroke="#888" label={{ value: 'Workers', position: 'bottom', fill: '#888' }} />
        <YAxis stroke="#888" label={{ value: 'Time (s)', angle: -90, position: 'insideLeft', fill: '#888' }} />
        <Tooltip contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: '8px' }} />
        <Legend />
        <Line type="monotone" dataKey="OpenMP" stroke="#22c55e" strokeWidth={3} dot={{ r: 5 }} />
        <Line type="monotone" dataKey="POSIX" stroke="#f97316" strokeWidth={3} dot={{ r: 5 }} />
        <Line type="monotone" dataKey="MPI" stroke="#a855f7" strokeWidth={3} dot={{ r: 5 }} />
      </LineChart>
    </ResponsiveContainer>
  )
}

function SpeedupChart() {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={speedupData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
        <XAxis dataKey="workers" stroke="#888" label={{ value: 'Workers', position: 'bottom', fill: '#888' }} />
        <YAxis stroke="#888" label={{ value: 'Speedup (×)', angle: -90, position: 'insideLeft', fill: '#888' }} />
        <Tooltip contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: '8px' }} />
        <Legend />
        <Line type="monotone" dataKey="Ideal" stroke="#555" strokeWidth={2} strokeDasharray="5 5" dot={false} />
        <Line type="monotone" dataKey="OpenMP" stroke="#22c55e" strokeWidth={3} dot={{ r: 5 }} />
        <Line type="monotone" dataKey="POSIX" stroke="#f97316" strokeWidth={3} dot={{ r: 5 }} />
        <Line type="monotone" dataKey="MPI" stroke="#a855f7" strokeWidth={3} dot={{ r: 5 }} />
      </LineChart>
    </ResponsiveContainer>
  )
}

function CudaChart() {
  return (
    <div className="h-full flex flex-col justify-center">
      <div className="overflow-x-auto">
        <table className="w-full text-center">
          <thead>
            <tr className="text-gray-400 border-b border-gray-700">
              <th className="p-4">Filter</th>
              <th className="p-4">Serial (s)</th>
              <th className="p-4">CUDA (s)</th>
              <th className="p-4">Speedup</th>
            </tr>
          </thead>
          <tbody>
            {cudaComparison.map((row) => (
              <tr key={row.filter} className="border-b border-gray-800">
                <td className="p-4 text-gray-300 font-medium">{row.filter}</td>
                <td className="p-4 text-red-400 font-mono">{row.Serial}</td>
                <td className="p-4 text-green-400 font-mono">{row.CUDA}</td>
                <td className="p-4">
                  <span className="bg-green-500/20 text-green-300 px-3 py-1 rounded-full font-bold">
                    {row.speedup}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-8 text-center">
        <p className="text-gray-400 text-sm">
          Tesla T4: 40 SMs • 2560 CUDA cores • 300 GB/s bandwidth
        </p>
        <p className="text-gray-400 text-sm mt-1">
          Shared memory tiling + constant memory broadcast = massive throughput
        </p>
      </div>
    </div>
  )
}

function EfficiencyChart() {
  const colors = ['#22c55e', '#f97316', '#a855f7', '#ef4444', '#06b6d4']
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={efficiencyData} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
        <XAxis dataKey="impl" stroke="#888" tick={{ fontSize: 11 }} interval={0} />
        <YAxis stroke="#888" label={{ value: 'Efficiency (%)', angle: -90, position: 'insideLeft', fill: '#888' }} />
        <Tooltip contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: '8px' }} />
        <Bar dataKey="efficiency" radius={[6, 6, 0, 0]}>
          {efficiencyData.map((_, i) => (
            <Cell key={i} fill={colors[i]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function RMSEChart() {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={rmseData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
        <XAxis dataKey="impl" stroke="#888" />
        <YAxis stroke="#888" label={{ value: 'RMSE', angle: -90, position: 'insideLeft', fill: '#888' }} />
        <Tooltip contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: '8px' }} />
        <Legend />
        <Bar dataKey="blur" fill="#3b82f6" name="Blur" radius={[4, 4, 0, 0]} />
        <Bar dataKey="edge" fill="#eab308" name="Edge" radius={[4, 4, 0, 0]} />
        <Bar dataKey="sharpen" fill="#ef4444" name="Sharpen" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}
