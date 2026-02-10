import { motion } from 'framer-motion';

const WhyAISection = () => {
  const reasons = [
    {
      num: '01',
      title: 'Objective Consistency',
      desc: 'Computer vision eliminates human bias, providing repeatable measurements with 95%+ accuracy across all evaluations'
    },
    {
      num: '02',
      title: 'Standardized Data',
      desc: 'Uniform measurement methodology creates reliable datasets for breeding programs and genetic research'
    },
    {
      num: '03',
      title: 'Scalable Solution',
      desc: 'Smartphone-based deployment enables evaluation anywhere, from remote villages to government centers'
    },
    {
      num: '04',
      title: 'Rapid Processing',
      desc: 'Sub-minute analysis compared to 25-30 minute manual assessment allows higher throughput screening'
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
    hidden: { opacity: 0, x: -20 },
    visible: {
      opacity: 1,
      x: 0,
      transition: {
        duration: 0.5
      }
    }
  };

  return (
    <section className="bg-green-50 py-24">
      <div className="max-w-6xl mx-auto px-6">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="mb-16"
        >
          <div className="text-sm font-bold tracking-widest uppercase mb-6 text-green-700">
            Why AI Classification
          </div>
          <h2 className="text-4xl md:text-5xl font-bold max-w-3xl text-gray-900 border-l-8 border-green-700 pl-6">
            The Case for Computer Vision in Livestock Assessment
          </h2>
        </motion.div>

        <motion.div 
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid md:grid-cols-2 gap-12"
        >
          {reasons.map((reason, index) => (
            <motion.div 
              key={reason.num} 
              variants={itemVariants}
              className={`border-l-8 pl-8 transition-colors bg-white shadow-lg p-6 hover:shadow-xl ${
                index % 4 === 0 ? 'border-orange-600 hover:border-orange-700' :
                index % 4 === 1 ? 'border-green-700 hover:border-green-800' :
                index % 4 === 2 ? 'border-orange-500 hover:border-orange-600' :
                'border-green-600 hover:border-green-700'
              }`}
            >
              <div className={`text-sm font-bold tracking-widest uppercase mb-4 ${
                index % 4 === 0 ? 'text-orange-600' :
                index % 4 === 1 ? 'text-green-700' :
                index % 4 === 2 ? 'text-orange-500' :
                'text-green-600'
              }`}>
                {reason.num}
              </div>
              <h3 className="text-2xl font-bold mb-4 text-gray-900">{reason.title}</h3>
              <p className="text-gray-700 leading-relaxed font-medium">{reason.desc}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default WhyAISection;
