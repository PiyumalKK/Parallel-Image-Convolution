import { motion } from 'framer-motion'

const navItems = [
  { id: 'hero', label: 'Home' },
  { id: 'convolution', label: 'Convolution' },
  { id: 'serial-vs-parallel', label: 'Overview' },
  { id: 'implementations', label: 'Implementations' },
  { id: 'demo', label: 'Demo' },
  { id: 'results', label: 'Results' },
  { id: 'conclusion', label: 'Conclusion' },
]

export default function Navigation() {
  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ delay: 0.5, type: 'spring', stiffness: 100 }}
      className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl bg-black/40 border-b border-white/10"
    >
      <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
        <span className="text-lg font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
          HPC Project - Group 42
        </span>
        <div className="hidden md:flex gap-6">
          {navItems.map((item) => (
            <a
              key={item.id}
              href={`#${item.id}`}
              className="text-sm text-gray-400 hover:text-white transition-colors duration-200"
            >
              {item.label}
            </a>
          ))}
        </div>
      </div>
    </motion.nav>
  )
}
