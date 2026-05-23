import Hero from './sections/Hero'
import ConvolutionDemo from './sections/ConvolutionDemo'
import SerialVsParallel from './sections/SerialVsParallel'
import SerialAnimation from './sections/SerialAnimation'
import OpenMPAnimation from './sections/OpenMPAnimation'
import PthreadsAnimation from './sections/PthreadsAnimation'
import MPIAnimation from './sections/MPIAnimation'
import CUDAAnimation from './sections/CUDAAnimation'
import HybridAnimation from './sections/HybridAnimation'
import DemoSection from './sections/DemoSection'
import Results from './sections/Results'
import Conclusion from './sections/Conclusion'
import Navigation from './components/Navigation'

function App() {
  return (
    <div className="relative">
      <Navigation />
      <Hero />
      <ConvolutionDemo />
      <SerialVsParallel />
      <div id="implementations">
        <SerialAnimation />
        <OpenMPAnimation />
        <PthreadsAnimation />
        <MPIAnimation />
        <CUDAAnimation />
        <HybridAnimation />
      </div>
      <DemoSection />
      <Results />
      <Conclusion />
    </div>
  )
}

export default App
