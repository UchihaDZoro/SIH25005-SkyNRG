import { motion } from 'framer-motion';

const CTASection = () => {
  return (
    <section className="relative py-32 overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-green-700 via-green-800 to-green-900"></div>
      <div className="relative max-w-5xl mx-auto px-6">
        <div className="text-center space-y-12">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-8 leading-tight text-white">
              Ready to Transform Livestock Evaluation?
            </h2>
            <p className="text-xl text-gray-200 max-w-3xl mx-auto leading-relaxed">
              Experience AI-powered classification. Upload images and receive 
              comprehensive type scores in under one minute.
            </p>
          </motion.div>

          <div className="flex flex-wrap justify-center gap-6">
            <a href="/classify">
              <motion.button 
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="px-12 py-5 bg-orange-600 text-white font-bold tracking-wide hover:bg-orange-700 transition-all text-lg shadow-xl hover:shadow-2xl border-b-4 border-orange-800"
              >
                TRY CLASSIFICATION
              </motion.button>
            </a>
            <a href="/breeds">
              <motion.button 
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="px-12 py-5 border-2 border-white text-white font-bold tracking-wide hover:bg-white hover:text-green-800 transition-all text-lg"
              >
                BROWSE BREEDS
              </motion.button>
            </a>
          </div>

          <motion.div 
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.4, duration: 1 }}
            className="pt-12 border-t-4 border-orange-500"
          >
            <div className="grid md:grid-cols-3 gap-12 text-center">
              <div>
                <div className="text-sm font-bold tracking-widest uppercase mb-3 text-orange-400">
                  Processing Speed
                </div>
                <div className="text-3xl font-bold text-white">&lt; 1 Minute</div>
              </div>
              <div>
                <div className="text-sm font-bold tracking-widest uppercase mb-3 text-orange-400">
                  Accuracy Rate
                </div>
                <div className="text-3xl font-bold text-white">95%+</div>
              </div>
              <div>
                <div className="text-sm font-bold tracking-widest uppercase mb-3 text-orange-400">
                  Breeds Supported
                </div>
                <div className="text-3xl font-bold text-white">56 Total</div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default CTASection;
