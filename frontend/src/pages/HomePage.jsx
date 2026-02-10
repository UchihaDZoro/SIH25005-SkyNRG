import HeroSection from '../components/home/HeroSection';
import ProblemSection from '../components/home/ProblemSection';
import WhyAISection from '../components/home/WhyAISection';
import ApproachSection from '../components/home/ApproachSection';
import SolutionSection from '../components/home/SolutionSection';
import TechnologySection from '../components/home/TechnologySection';
import CTASection from '../components/home/CTASection';

const HomePage = () => {
  return (
    <div>
      <HeroSection />
      <ProblemSection />
      <WhyAISection />
      <ApproachSection />
      <SolutionSection />
      <TechnologySection />
      <CTASection />
    </div>
  );
};

export default HomePage;
