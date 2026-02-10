import { motion } from 'framer-motion';

const SolutionSection = () => {
  const features = [
    {
      title: 'Pose Estimation Technology',
      desc: 'YOLO11m-pose deep learning model trained for livestock anatomy, detecting skeletal keypoints with research-grade accuracy'
    },
    {
      title: 'Breed-Specific Scoring',
      desc: 'Customized evaluation criteria for indigenous cattle and buffalo breeds based on official type classification standards'
    },
    {
      title: 'Mobile-First Design',
      desc: 'Progressive web application enabling field evaluation without specialized equipment or stable connectivity'
    },
    {
      title: 'Data Integration',
      desc: 'Structured output compatible with national breeding databases, supporting RGM analytics and conservation tracking'
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
    hidden: { opacity: 0, scale: 0.9 },
    visible: {
      opacity: 1,
      scale: 1,
      transition: {
        duration: 0.5
      }
    }
  };

  return (
    <section className="bg-orange-50 py-24">
      <div className="max-w-6xl mx-auto px-6">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="mb-16"
        >
          <div className="text-sm font-bold tracking-widest uppercase mb-6 text-orange-700">
            Our Solution
          </div>
          <h2 className="text-4xl md:text-5xl font-bold mb-6 max-w-3xl text-gray-900 border-l-8 border-orange-600 pl-6">
            AI-Powered Classification System
          </h2>
          <p className="text-xl text-gray-700 max-w-3xl leading-relaxed pl-6">
            Combining computer vision, breed science, and mobile technology to deliver 
            accessible, objective livestock evaluation.
          </p>
        </motion.div>

        <motion.div 
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid md:grid-cols-2 gap-12"
        >
          {features.map((feature, index) => (
            <motion.div 
              key={index} 
              variants={itemVariants}
              className={`border-l-8 pl-8 transition-colors bg-white shadow-lg p-6 hover:shadow-xl ${
                index % 4 === 0 ? 'border-orange-600 hover:border-orange-700' :
                index % 4 === 1 ? 'border-green-600 hover:border-green-700' :
                index % 4 === 2 ? 'border-orange-500 hover:border-orange-600' :
                'border-green-700 hover:border-green-800'
              }`}
            >
              <h3 className="text-2xl font-bold mb-4 text-gray-900">{feature.title}</h3>
              <p className="text-gray-700 leading-relaxed font-medium">{feature.desc}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default SolutionSection;
