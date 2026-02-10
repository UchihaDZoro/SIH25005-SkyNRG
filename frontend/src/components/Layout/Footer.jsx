import { Link } from 'react-router-dom';
import { FiMail, FiPhone, FiMapPin, FiGithub, FiLinkedin, FiTwitter, FiInstagram, FiSend } from 'react-icons/fi';
import { useState } from 'react';

const Footer = () => {
  const [email, setEmail] = useState('');
  const currentYear = new Date().getFullYear();

  const handleNewsletterSubmit = (e) => {
    e.preventDefault();
    alert('Thank you for subscribing!');
    setEmail('');
  };

  const footerSections = {
    product: {
      title: 'Product',
      links: [
        { name: 'Classification System', path: '/classify' },
        { name: 'Breed Database', path: '/breeds' },
        { name: 'Image Gallery', path: '/gallery' },
        { name: 'Type Scoring', path: '/docs' },
        { name: 'How It Works', path: '/docs' },
      ]
    },
    company: {
      title: 'Company',
      links: [
        { name: 'About Us', path: '/about' },
        { name: 'Rashtriya Gokul Mission', path: '/rgm' },
        { name: 'Our Mission', path: '/about' },
        { name: 'Contact Us', path: '/contact' },
        { name: 'Careers', path: '/contact' },
      ]
    },
    resources: {
      title: 'Resources',
      links: [
        { name: 'Documentation', path: '/docs' },
        { name: 'API Reference', path: '/api' },
        { name: 'FAQs', path: '/faq' },
        { name: 'User Guide', path: '/docs' },
        { name: 'Support Center', path: '/contact' },
      ]
    },
    developers: {
      title: 'Developers',
      links: [
        { name: 'API Documentation', path: '/api' },
        { name: 'GitHub Repository', path: '#' },
        { name: 'Code Examples', path: '/api' },
        { name: 'Integration Guide', path: '/docs' },
        { name: 'Developer Forum', path: '#' },
      ]
    },
    legal: {
      title: 'Legal',
      links: [
        { name: 'Privacy Policy', path: '#' },
        { name: 'Terms of Service', path: '#' },
        { name: 'Cookie Policy', path: '#' },
        { name: 'Data Protection', path: '#' },
        { name: 'Acceptable Use', path: '#' },
      ]
    }
  };

  const stats = [
    { number: '95%+', label: 'Accuracy Rate' },
    { number: '56', label: 'Breeds Supported' },
    { number: '<1min', label: 'Processing Time' },
    { number: '24/7', label: 'Availability' }
  ];

  return (
    <footer className="bg-white">
      
      {/* Newsletter Section with Gradient */}
      <div className="bg-gradient-to-r from-orange-50 via-white to-green-50 border-t-4 border-orange-600">
        <div className="max-w-7xl mx-auto px-6 py-16">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <div className="text-sm font-bold tracking-widest uppercase mb-4 text-orange-600">
                Newsletter
              </div>
              <h3 className="text-3xl md:text-4xl font-bold mb-4 text-gray-900">
                Stay Updated with Latest Developments
              </h3>
              <p className="text-gray-600 leading-relaxed font-medium">
                Get insights on AI-powered livestock classification, breed conservation updates, 
                and technical improvements delivered to your inbox.
              </p>
            </div>
            
            <form onSubmit={handleNewsletterSubmit} className="flex flex-col sm:flex-row gap-3">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Enter your email address"
                required
                className="flex-1 px-6 py-4 bg-white border-2 border-gray-300 text-gray-900 placeholder-gray-500 focus:outline-none focus:border-orange-600 transition-colors font-medium"
              />
              <button
                type="submit"
                className="px-8 py-4 bg-orange-600 text-white font-bold tracking-wide hover:bg-orange-700 transition-all flex items-center justify-center gap-2 shadow-lg hover:shadow-xl border-b-4 border-orange-800"
              >
                <FiSend className="w-5 h-5" />
                SUBSCRIBE
              </button>
            </form>
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="border-b-4 border-green-600 bg-gray-50">
        <div className="max-w-7xl mx-auto px-6 py-16">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className={`text-4xl md:text-5xl font-bold mb-2 ${
                  index === 0 ? 'text-orange-600' :
                  index === 1 ? 'text-green-600' :
                  index === 2 ? 'text-orange-500' :
                  'text-green-700'
                }`}>{stat.number}</div>
                <div className="text-sm text-gray-600 tracking-wide uppercase font-bold">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Footer Content */}
      <div className="max-w-7xl mx-auto px-6 py-20 bg-white">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-12 gap-12 mb-16">
          
          {/* Brand & Contact Section */}
          <div className="lg:col-span-4">
            <div className="mb-8">
              <div className="text-2xl font-bold tracking-tight mb-2 text-gray-900">
                LIVESTOCK
              </div>
              <div className="flex items-center gap-2 mb-6">
                <div className="w-8 h-1 bg-orange-600"></div>
                <span className="text-[10px] font-bold text-green-700 tracking-[0.3em] uppercase">
                  Classification
                </span>
                <div className="w-8 h-1 bg-green-600"></div>
              </div>
              <p className="text-gray-600 leading-relaxed mb-8 font-medium">
                AI-powered livestock classification system supporting indigenous breed 
                conservation through objective, scientific evaluation methodology.
              </p>
            </div>

            {/* Contact Information */}
            <div className="space-y-4 mb-8">
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 bg-orange-100 flex items-center justify-center border-2 border-orange-600">
                  <FiMail className="w-5 h-5 text-orange-600" />
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-1 font-bold uppercase">Email</div>
                  <a href="mailto:info@livestockclassification.com" className="text-gray-900 hover:text-orange-600 transition-colors font-medium">
                    info@livestockclassification.com
                  </a>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 bg-green-100 flex items-center justify-center border-2 border-green-600">
                  <FiPhone className="w-5 h-5 text-green-600" />
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-1 font-bold uppercase">Phone</div>
                  <a href="tel:+911234567890" className="text-gray-900 hover:text-green-600 transition-colors font-medium">
                    +91 12345 67890
                  </a>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 bg-orange-100 flex items-center justify-center border-2 border-orange-600">
                  <FiMapPin className="w-5 h-5 text-orange-600" />
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-1 font-bold uppercase">Location</div>
                  <span className="text-gray-900 font-medium">New Delhi, India</span>
                </div>
              </div>
            </div>

            {/* Social Links */}
            <div>
              <div className="text-sm font-bold tracking-widest uppercase mb-4 text-gray-700">
                Follow Us
              </div>
              <div className="flex gap-3">
                <a
                  href="#"
                  className="w-12 h-12 border-2 border-orange-600 flex items-center justify-center hover:bg-orange-600 hover:text-white transition-all text-gray-900"
                >
                  <FiGithub className="w-5 h-5" />
                </a>
                <a
                  href="#"
                  className="w-12 h-12 border-2 border-green-600 flex items-center justify-center hover:bg-green-600 hover:text-white transition-all text-gray-900"
                >
                  <FiLinkedin className="w-5 h-5" />
                </a>
                <a
                  href="#"
                  className="w-12 h-12 border-2 border-orange-600 flex items-center justify-center hover:bg-orange-600 hover:text-white transition-all text-gray-900"
                >
                  <FiTwitter className="w-5 h-5" />
                </a>
                <a
                  href="#"
                  className="w-12 h-12 border-2 border-green-600 flex items-center justify-center hover:bg-green-600 hover:text-white transition-all text-gray-900"
                >
                  <FiInstagram className="w-5 h-5" />
                </a>
              </div>
            </div>
          </div>

          {/* Links Sections */}
          {Object.entries(footerSections).map(([key, section], idx) => (
            <div key={key} className="lg:col-span-2">
              <h4 className={`text-sm font-bold tracking-widest uppercase mb-6 pb-2 border-b-2 ${
                idx % 2 === 0 ? 'text-orange-600 border-orange-600' : 'text-green-700 border-green-600'
              }`}>
                {section.title}
              </h4>
              <ul className="space-y-3">
                {section.links.map((link, index) => (
                  <li key={index}>
                    <Link
                      to={link.path}
                      className="text-gray-700 hover:text-orange-600 transition-colors font-medium text-sm block hover:translate-x-1 transition-transform"
                    >
                      {link.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t-4 border-gray-200">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <div className="text-sm text-gray-600 text-center md:text-left">
              <p className="mb-2 font-medium">
                © {currentYear} Livestock Classification System. All rights reserved.
              </p>
              <p className="text-xs text-gray-500 font-medium">
                Smart India Hackathon 2025 | Powered by AI & Computer Vision
              </p>
            </div>
            
            <div className="flex flex-wrap gap-6 text-sm">
              <Link to="#" className="text-gray-600 hover:text-orange-600 transition-colors font-medium">
                Accessibility
              </Link>
              <Link to="#" className="text-gray-600 hover:text-orange-600 transition-colors font-medium">
                Sitemap
              </Link>
              <Link to="/contact" className="text-gray-600 hover:text-orange-600 transition-colors font-medium">
                Report Issue
              </Link>
              <span className="px-3 py-1 bg-gray-200 text-xs tracking-wide font-bold border-2 border-gray-300">
                v1.0.0
              </span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
