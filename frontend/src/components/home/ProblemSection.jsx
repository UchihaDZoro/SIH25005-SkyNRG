import { motion } from 'framer-motion';

const ProblemSection = () => {
  const problems = [
    {
      title: 'Subjective Assessment',
      desc: 'Manual evaluation leads to 35% inter-observer variability, directly compromising breed selection decisions and threatening indigenous cattle conservation programs'
    },
    {
      title: 'Time Intensive Process',
      desc: '25-30 minutes per animal creates severe bottlenecks - limiting large-scale breeding programs to only 15-20 evaluations per day per expert'
    },
    {
      title: 'Limited Rural Access',
      desc: 'Expert evaluators concentrated in urban areas, leaving over 70% of remote breeding centers without scientific assessment capabilities'
    },
    {
      title: 'Inconsistent Standards',
      desc: 'Regional variations in breed standards interpretation reduce genetic database quality by up to 40%, hindering national improvement programs'
    }
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5
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
            The Challenge
          </div>
          <h2 className="text-4xl md:text-5xl font-bold mb-6 max-w-3xl text-gray-900 border-l-8 border-orange-600 pl-6">
            Current Livestock Evaluation Limitations
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl leading-relaxed pl-6">
            Traditional manual type scoring faces critical challenges that hinder effective 
            indigenous breed conservation and genetic improvement programs.
          </p>
        </motion.div>

        <motion.div 
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid md:grid-cols-2 gap-12"
        >
          {problems.map((problem, index) => (
            <motion.div 
              key={index} 
              variants={itemVariants}
              className={`border-l-8 pl-8 transition-colors shadow-lg p-6 bg-gray-50 hover:shadow-xl ${
                index % 4 === 0 ? 'border-orange-600 hover:border-orange-700' :
                index % 4 === 1 ? 'border-green-600 hover:border-green-700' :
                index % 4 === 2 ? 'border-orange-500 hover:border-orange-600' :
                'border-green-700 hover:border-green-800'
              }`}
            >
              <h3 className="text-2xl font-bold mb-4 text-gray-900">{problem.title}</h3>
              <p className="text-gray-700 leading-relaxed font-medium">{problem.desc}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default ProblemSection;
