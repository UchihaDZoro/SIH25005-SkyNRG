import { motion } from 'framer-motion';

const TechnologySection = () => {
  const stack = [
    {
      category: 'Computer Vision',
      items: ['YOLO11m-pose Estimation', 'PyTorch Framework', 'OpenCV Processing']
    },
    {
      category: 'Backend',
      items: ['FastAPI Python', 'Pydantic Validation', 'Async Processing']
    },
    {
      category: 'Frontend',
      items: ['React 18', 'Vite Build Tool', 'Tailwind CSS']
    },
    {
      category: 'Deployment',
      items: ['Progressive Web App', 'Mobile Optimization', 'API Integration']
    }
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5,
        ease: "easeOut"
      }
    }
  };

  return (
    <section className="bg-white py-24">
      <div className="max-w-6xl mx-auto px-6">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="mb-16"
        >
          <div className="text-sm font-bold tracking-widest uppercase mb-6 text-orange-600">
            Technology Stack
          </div>
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 border-l-8 border-green-700 pl-6">
            Built with Modern Tools
          </h2>
        </motion.div>

        <motion.div 
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid md:grid-cols-4 gap-8"
        >
          {stack.map((section, index) => (
            <motion.div 
              key={index} 
              variants={itemVariants}
              className={`border-t-8 pt-6 hover:scale-105 transition-all shadow-lg p-6 bg-gray-50 ${
                index === 0 ? 'border-orange-600 hover:border-orange-700' :
                index === 1 ? 'border-green-700 hover:border-green-800' :
                index === 2 ? 'border-orange-500 hover:border-orange-600' :
                'border-green-600 hover:border-green-700'
              }`}
            >
              <h3 className={`text-sm font-bold tracking-widest uppercase mb-6 ${
                index === 0 ? 'text-orange-600' :
                index === 1 ? 'text-green-700' :
                index === 2 ? 'text-orange-500' :
                'text-green-600'
              }`}>
                {section.category}
              </h3>
              <ul className="space-y-3">
                {section.items.map((item, i) => (
                  <li key={i} className="text-lg font-medium text-gray-800">{item}</li>
                ))}
              </ul>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default TechnologySection;
