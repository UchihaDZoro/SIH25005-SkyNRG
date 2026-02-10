import { TypeAnimation } from 'react-type-animation';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

const HeroSection = () => {
  return (
    <section className="min-h-screen bg-gradient-to-br from-orange-600 via-orange-700 to-amber-800 text-white flex items-center relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 opacity-10">
        <motion.div 
          animate={{ 
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.5, 0.3], 
          }}
          transition={{ 
            duration: 8, 
            repeat: Infinity,
            ease: "easeInOut" 
          }}
          className="absolute top-20 left-10 w-72 h-72 bg-white blur-3xl"
        ></motion.div>
        <motion.div 
          animate={{ 
            scale: [1, 1.5, 1],
            opacity: [0.3, 0.6, 0.3], 
          }}
          transition={{ 
            duration: 10, 
            repeat: Infinity, 
            ease: "easeInOut",
            delay: 1
          }}
          className="absolute bottom-20 right-10 w-96 h-96 bg-green-600 blur-3xl"
        ></motion.div>
      </div>
      <div className="max-w-7xl mx-auto px-6 py-32 w-full relative z-10">
        <div className="grid lg:grid-cols-2 gap-16 lg:gap-24 items-center">
          
          {/* Left Side - Main Content */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.6 }}
              className="inline-flex items-center gap-2 px-4 py-2 bg-green-600 text-white text-sm font-semibold tracking-widest uppercase mb-8 border-2 border-green-500"
            >
              <span className="w-2 h-2 bg-white animate-pulse"></span>
              Smart India Hackathon 2025
            </motion.div>
            
            <motion.h1 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.6 }}
              className="text-5xl md:text-6xl lg:text-7xl font-light mb-8 leading-tight"
            >
              AI-Powered Livestock Classification
            </motion.h1>
            
            <motion.p 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6, duration: 0.6 }}
              className="text-lg md:text-xl text-orange-100 leading-relaxed mb-12"
            >
              Objective evaluation through computer vision. Supporting Rashtriya Gokul Mission's 
              conservation goals with standardized, scientific breed assessment.
            </motion.p>
            
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8, duration: 0.6 }}
              className="flex flex-wrap gap-6"
            >
              <Link 
                to="/classify"
                className="inline-block px-10 py-4 bg-white text-orange-700 font-bold tracking-wide hover:bg-green-600 hover:text-white transition-all shadow-xl border-b-4 border-orange-800 hover:border-green-800"
              >
                START CLASSIFICATION
              </Link>
              <Link 
                to="/about"
                className="inline-block px-10 py-4 border-2 border-white text-white font-medium tracking-wide hover:bg-white hover:text-orange-700 transition-all"
              >
                LEARN MORE
              </Link>
            </motion.div>
          </motion.div>

          {/* Right Side - Large Typewriter Text */}
          <motion.div 
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, ease: "easeOut", delay: 0.2 }}
            className="relative"
          >
            <div className="text-right">
              <div className="text-5xl md:text-6xl lg:text-7xl xl:text-8xl font-light leading-tight">
                <TypeAnimation
                  sequence={[
                    'Precision\nMeets\nTradition',
                    3000,
                    'Science\nSupports\nConservation',
                    3000,
                    'Technology\nPreserves\nHeritage',
                    3000,
                    'Data\nDrives\nBreeding',
                    3000,
                  ]}
                  wrapper="span"
                  speed={50}
                  style={{ 
                    whiteSpace: 'pre-line',
                    display: 'inline-block',
                    color: 'white'
                  }}
                  repeat={Infinity}
                />
              </div>
              
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1.2, duration: 1 }}
                className="mt-12 text-sm font-medium tracking-widest uppercase text-orange-200"
              >
                Honest AI. Real Impact.
              </motion.div>
            </div>
          </motion.div>

        </div>
      </div>
    </section>
  );
};

export default HeroSection;
