import { motion } from 'framer-motion';

const ApproachSection = () => {
  const steps = [
    {
      num: '01',
      title: 'Image Capture',
      desc: 'Three standardized photographs using smartphone camera: Left side view, Right side view, and Rear/Udder view for comprehensive coverage'
    },
    {
      num: '02',
      title: 'Keypoint Detection',
      desc: 'YOLO11m-pose estimation identifies 30+ anatomical landmarks with precision, creating a digital skeletal map of the animal'
    },
    {
      num: '03',
      title: 'Measurement Extraction',
      desc: 'Computer vision algorithms calculate body proportions, angles, and ratios from detected keypoints using standardized methodology'
    },
    {
      num: '04',
      title: 'Type Score Generation',
      desc: 'Measurements mapped to breed-specific standards, generating objective type scores and detailed evaluation reports'
    },
    {
      num: '05',
      title: 'Milk Yield Prediction',
      desc: 'Advanced analytics predict daily and lactation milk yield based on body measurements, udder characteristics, and breed parameters'
    },
    {
      num: '06',
      title: 'Longevity & Health Assessment',
      desc: 'Predictive models estimate productive lifespan, reproduction efficiency, and overall health indicators for informed breeding decisions'
    }
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.3
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, x: -30 },
    visible: {
      opacity: 1,
      x: 0,
      transition: {
        duration: 0.6,
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
          <div className="text-sm font-bold tracking-widest uppercase mb-6 text-green-700">
            Our Methodology
          </div>
          <h2 className="text-4xl md:text-5xl font-bold mb-6 max-w-3xl text-gray-900 border-l-8 border-green-700 pl-6">
            Six-Step Classification Process
          </h2>
          <p className="text-xl text-gray-700 max-w-3xl leading-relaxed pl-6">
            Systematic approach combining computer vision with established livestock 
            evaluation standards for accurate, repeatable results and actionable insights.
          </p>
        </motion.div>

        <motion.div 
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="space-y-8"
        >
          {steps.map((step, index) => (
            <motion.div 
              key={step.num} 
              variants={itemVariants}
              className={`border-l-8 pt-8 pl-8 pb-8 transition-colors shadow-lg hover:shadow-xl ${
                index % 3 === 0 ? 'border-orange-600 hover:border-orange-700 bg-orange-50' :
                index % 3 === 1 ? 'border-green-700 hover:border-green-800 bg-green-50' :
                'border-orange-500 hover:border-orange-600 bg-gray-50'
              }`}
            >
              <div className="grid md:grid-cols-12 gap-8">
                <div className="md:col-span-2">
                  <div className={`text-5xl font-bold ${
                    index % 3 === 0 ? 'text-orange-600' :
                    index % 3 === 1 ? 'text-green-700' :
                    'text-orange-500'
                  }`}>
                    {step.num}
                  </div>
                </div>
                <div className="md:col-span-10">
                  <h3 className="text-2xl font-bold mb-4 text-gray-900">{step.title}</h3>
                  <p className="text-gray-700 leading-relaxed text-lg font-medium">{step.desc}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default ApproachSection;
