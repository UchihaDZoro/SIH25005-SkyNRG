import { useState, useEffect } from 'react';
import { FiHash, FiServer, FiCode, FiCopy, FiCheck, FiChevronRight, FiBox, FiImage, FiCpu, FiDatabase, FiTrash2, FiSearch, FiMenu, FiX } from 'react-icons/fi';

const APIPage = () => {
  const [activeSection, setActiveSection] = useState('introduction');
  const [copiedId, setCopiedId] = useState(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Handle scroll spy to update active section
  useEffect(() => {
    const handleScroll = () => {
      const sections = ['introduction', 'authentication', ...endpoints.map(e => e.id)];
      for (const section of sections) {
        const element = document.getElementById(section);
        if (element) {
          const rect = element.getBoundingClientRect();
          if (rect.top >= 0 && rect.top <= 300) {
            setActiveSection(section);
            break;
          }
        }
      }
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleCopy = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const scrollToSection = (id) => {
    setActiveSection(id);
    setMobileMenuOpen(false);
    const element = document.getElementById(id);
    if (element) {
      const offset = 100; // Adjust for sticky header
      const elementPosition = element.getBoundingClientRect().top;
      const offsetPosition = elementPosition + window.pageYOffset - offset;
      window.scrollTo({
        top: offsetPosition,
        behavior: "smooth"
      });
    }
  };

  const endpoints = [
    {
      id: 'create-classification',
      method: 'POST',
      path: '/classification/create',
      title: 'Create Classification',
      description: 'Initialize a new classification record. Animal info is optional - send empty object for auto-generated defaults.',
      color: 'emerald',
      icon: FiBox,
      params: [],
      body: {},
      response: {
        success: true,
        message: "Classification created",
        data: {
          id: "656a1b2c3d4e5f6g7h8i9j0k",
          status: "created",
          animalInfo: {
            tagNumber: "AUTO-1733684425",
            animalType: "cattle",
            breed: "Unknown",
            village: "Not specified",
            farmerName: "Unknown"
          }
        }
      }
    },
    {
      id: 'upload-images',
      method: 'POST',
      path: '/classification/{id}/upload-images',
      title: 'Upload Images',
      description: 'Upload exactly 4 standardized images for the classification (Top, Rear, Side, Bottom views).',
      color: 'blue',
      icon: FiImage,
      params: [
        { name: 'id', type: 'string', required: true, desc: 'Classification ID' }
      ],
      body: 'FormData: images (4 files)',
      response: {
        success: true,
        message: "Images uploaded successfully",
        data: [
          { filename: "...", url: "/uploads/...", angle: "top_view", status: "uploaded" },
          { filename: "...", url: "/uploads/...", angle: "rear_view", status: "uploaded" },
          { filename: "...", url: "/uploads/...", angle: "side_view", status: "uploaded" },
          { filename: "...", url: "/uploads/...", angle: "bottom_view", status: "uploaded" }
        ]
      }
    },
    {
      id: 'process-classification',
      method: 'POST',
      path: '/classification/{id}/process',
      title: 'Process Classification',
      description: 'Trigger AI analysis to evaluate 20 traits based on uploaded images.',
      color: 'violet',
      icon: FiCpu,
      params: [
        { name: 'id', type: 'string', required: true, desc: 'Classification ID' }
      ],
      response: {
        success: true,
        message: "Classification completed using official Type Evaluation Format",
        data: {
          id: "656a1b2c3d4e5f6g7h8i9j0k",
          status: "completed",
          totalTraits: 20
        }
      }
    },
    {
      id: 'get-results',
      method: 'GET',
      path: '/classification/{id}/results',
      title: 'Get Results',
      description: 'Retrieve classification results with trait scores, section scores, and overall grade.',
      color: 'amber',
      icon: FiDatabase,
      params: [
        { name: 'id', type: 'string', required: true, desc: 'Classification ID' }
      ],
      response: {
        success: true,
        data: {
          id: "656a1b2c3d4e5f6g7h8i9j0k",
          status: "completed",
          overallScore: 6.7,
          grade: "Good",
          totalTraits: 20,
          confidenceLevel: "High",
          createdAt: "2025-12-08T15:30:00Z",
          categoryScores: { 
            "Body Capacity": 7.2, 
            "Dairy Character": 6.5,
            "Mammary System": 6.8,
            "Feet and Legs": 6.4,
            "Overall Appearance": 7.0
          },
          officialFormat: { 
            sections: {
              "Body Capacity": [
                { trait: "Stature", score: 7, measurement: "142 cm" },
                { trait: "Chest Width", score: 8, measurement: null }
              ]
            }
          }
        }
      }
    },
    {
      id: 'get-archive',
      method: 'GET',
      path: '/classification/archive',
      title: 'Get Archive',
      description: 'Retrieve paginated and filtered list of completed classifications.',
      color: 'cyan',
      icon: FiSearch,
      params: [
        { name: 'skip', type: 'integer', required: false, desc: 'Number of records to skip (default: 0)' },
        { name: 'limit', type: 'integer', required: false, desc: 'Number of records to return (default: 20)' },
        { name: 'grade', type: 'string', required: false, desc: 'Filter by grade (Excellent/Good/Fair/Poor)' },
        { name: 'search', type: 'string', required: false, desc: 'Search by classification ID' }
      ],
      response: {
        success: true,
        data: {
          results: [
            {
              id: "656a1b2c3d4e5f6g7h8i9j0k",
              overallScore: 6.7,
              grade: "Good",
              confidenceLevel: "High",
              createdAt: "2025-12-08T15:30:00Z"
            }
          ],
          total: 150,
          page: 1,
          totalPages: 8
        }
      }
    },
    {
      id: 'delete-classification',
      method: 'DELETE',
      path: '/classification/{id}',
      title: 'Delete Classification',
      description: 'Permanently remove a classification record.',
      color: 'rose',
      icon: FiTrash2,
      params: [
        { name: 'id', type: 'string', required: true, desc: 'Classification ID' }
      ],
      response: {
        success: true,
        message: "Classification deleted successfully",
        data: { id: "656a1b2c3d4e5f6g7h8i9j0k" }
      }
    }
  ];

  const MethodBadge = ({ method }) => {
    const colors = {
      GET: 'bg-blue-500/10 text-blue-600 border-blue-200',
      POST: 'bg-emerald-500/10 text-emerald-600 border-emerald-200',
      PUT: 'bg-amber-500/10 text-amber-600 border-amber-200',
      DELETE: 'bg-rose-500/10 text-rose-600 border-rose-200'
    };
    return (
      <span className={`px-3 py-1 rounded-full text-xs font-bold border ${colors[method] || 'bg-gray-100'}`}>
        {method}
      </span>
    );
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans selection:bg-orange-100 selection:text-orange-900 pt-20">
      
      {/* Background Decor */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-0 left-0 w-[800px] h-[800px] bg-orange-200/20 rounded-full blur-3xl -translate-x-1/2 -translate-y-1/2"></div>
        <div className="absolute bottom-0 right-0 w-[600px] h-[600px] bg-blue-200/20 rounded-full blur-3xl translate-x-1/3 translate-y-1/3"></div>
      </div>

      {/* Hero Header */}
      <div className="relative z-10 bg-white/50 backdrop-blur-xl border-b border-white/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-12 lg:py-24">
          <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-8">
            <div className="max-w-3xl">
              <div className="flex items-center gap-3 mb-6">
                <span className="px-3 py-1 rounded-full bg-orange-100 text-orange-700 text-xs font-bold tracking-wider uppercase border border-orange-200">
                  v1.0.0 Public Beta
                </span>
                <span className="flex items-center gap-2 text-slate-500 text-sm font-medium">
                  <FiServer className="w-4 h-4" /> REST API
                </span>
              </div>
              <h1 className="text-4xl lg:text-6xl font-extrabold text-slate-900 tracking-tight mb-6">
                Build with <span className="text-transparent bg-clip-text bg-gradient-to-r from-orange-600 to-amber-600">Confidence</span>
              </h1>
              <p className="text-lg lg:text-xl text-slate-600 leading-relaxed max-w-2xl">
                Integrate our AI-powered livestock evaluation system directly into your applications. 
                Standardized, reliable, and built for scale.
              </p>
            </div>
            {/* Abstract Code Visual - Hidden on mobile for better space */}
            <div className="hidden lg:block w-full max-w-md">
              <div className="bg-slate-900 rounded-2xl p-6 shadow-2xl border border-slate-800 transform rotate-3 hover:rotate-0 transition-transform duration-500">
                <div className="flex gap-2 mb-4">
                  <div className="w-3 h-3 rounded-full bg-red-500"></div>
                  <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                  <div className="w-3 h-3 rounded-full bg-green-500"></div>
                </div>
                <div className="space-y-2 font-mono text-xs">
                  <div className="text-slate-400">// Initialize client</div>
                  <div className="text-purple-400">const <span className="text-blue-400">api</span> = <span className="text-yellow-400">new</span> <span className="text-emerald-400">Client</span>({'{'}</div>
                  <div className="pl-4 text-slate-300">apiKey: <span className="text-orange-400">"sk_live_..."</span></div>
                  <div className="text-purple-400">{'}'});</div>
                  <div className="h-2"></div>
                  <div className="text-slate-400">// Classify animal</div>
                  <div className="text-purple-400">await <span className="text-blue-400">api</span>.<span className="text-blue-300">classify</span>({'{'}</div>
                  <div className="pl-4 text-slate-300">image: <span className="text-orange-400">file</span>,</div>
                  <div className="pl-4 text-slate-300">type: <span className="text-orange-400">"cattle"</span></div>
                  <div className="text-purple-400">{'}'});</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 py-8 lg:py-12">
        <div className="grid lg:grid-cols-12 gap-8 lg:gap-12">
          
          {/* Mobile Navigation (Accordion Style) */}
          <div className="lg:hidden mb-8">
            <button 
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="w-full flex items-center justify-between p-4 bg-white rounded-xl shadow-sm border border-slate-200 text-slate-900 font-bold"
            >
              <span className="flex items-center gap-2">
                <FiMenu className="w-5 h-5" /> Table of Contents
              </span>
              <FiChevronRight className={`w-5 h-5 transition-transform ${mobileMenuOpen ? 'rotate-90' : ''}`} />
            </button>
            
            {mobileMenuOpen && (
              <div className="mt-2 bg-white rounded-xl shadow-lg border border-slate-100 overflow-hidden p-4 space-y-6 animate-in fade-in slide-in-from-top-2">
                <div>
                  <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">Getting Started</h3>
                  <div className="grid grid-cols-1 gap-2">
                    {['introduction', 'authentication'].map((item) => (
                      <button 
                        key={item}
                        onClick={() => scrollToSection(item)}
                        className="text-left px-3 py-2 rounded-lg text-sm font-medium text-slate-600 hover:bg-slate-50 hover:text-orange-600 transition-colors"
                      >
                        {item.charAt(0).toUpperCase() + item.slice(1)}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">Endpoints</h3>
                  <div className="grid grid-cols-1 gap-2">
                    {endpoints.map((endpoint) => (
                      <button 
                        key={endpoint.id}
                        onClick={() => scrollToSection(endpoint.id)}
                        className="flex items-center gap-3 text-left px-3 py-2 rounded-lg text-sm font-medium text-slate-600 hover:bg-slate-50 hover:text-slate-900 transition-colors"
                      >
                        <span className={`w-2 h-2 rounded-full bg-${endpoint.color}-500`}></span>
                        {endpoint.title}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Desktop Sidebar Navigation */}
          <div className="hidden lg:block lg:col-span-3">
            <div className="sticky top-32 space-y-8">
              <div>
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">
                  Getting Started
                </h3>
                <ul className="space-y-1">
                  {['introduction', 'authentication'].map((item) => (
                    <li key={item}>
                      <button 
                        onClick={() => scrollToSection(item)}
                        className={`
                          w-full text-left px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200
                          ${activeSection === item 
                            ? 'bg-orange-50 text-orange-700 translate-x-2' 
                            : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'}
                        `}
                      >
                        {item.charAt(0).toUpperCase() + item.slice(1)}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">
                  Endpoints
                </h3>
                <ul className="space-y-1">
                  {endpoints.map((endpoint) => (
                    <li key={endpoint.id}>
                      <button 
                        onClick={() => scrollToSection(endpoint.id)}
                        className={`
                          group w-full flex items-center gap-3 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200
                          ${activeSection === endpoint.id 
                            ? 'bg-white shadow-md text-slate-900 translate-x-2 ring-1 ring-slate-200' 
                            : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'}
                        `}
                      >
                        <span className={`
                          w-1.5 h-1.5 rounded-full transition-all duration-300
                          ${activeSection === endpoint.id ? `bg-${endpoint.color}-500 scale-150` : 'bg-slate-300 group-hover:bg-slate-400'}
                        `}></span>
                        {endpoint.title}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-9 space-y-16 lg:space-y-24 pb-24">
            
            {/* Introduction Section */}
            <section id="introduction" className="scroll-mt-32">
              <div className="bg-white rounded-3xl p-6 lg:p-8 shadow-sm border border-slate-100">
                <div className="flex items-center gap-4 mb-6">
                  <div className="p-3 bg-slate-900 text-white rounded-2xl shadow-lg shadow-slate-900/20">
                    <FiHash className="w-6 h-6" />
                  </div>
                  <h2 className="text-2xl lg:text-3xl font-bold text-slate-900">Introduction</h2>
                </div>
                <div className="prose prose-slate prose-lg max-w-none">
                  <p className="text-slate-600">
                    The Animal Classification API provides a robust RESTful interface for the AI-powered livestock evaluation system. 
                    Designed for developers, it follows the official <strong className="text-slate-900">Type Evaluation Format (Annex II)</strong> standards for indigenous breeds.
                  </p>
                  <div className="grid md:grid-cols-2 gap-6 mt-8">
                    <div className="p-5 lg:p-6 bg-slate-50 rounded-2xl border border-slate-100">
                      <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Base URL</h4>
                      <div className="flex items-center gap-3">
                        <code className="flex-1 px-4 py-3 bg-white text-slate-700 rounded-xl text-xs lg:text-sm font-mono border border-slate-200 shadow-sm overflow-x-auto whitespace-nowrap">
                          http://localhost:8000/api/v1
                        </code>
                        <button 
                          onClick={() => handleCopy('http://localhost:8000/api/v1', 'base-url')}
                          className="p-3 bg-white text-slate-400 hover:text-orange-600 rounded-xl border border-slate-200 shadow-sm transition-colors flex-shrink-0"
                        >
                          {copiedId === 'base-url' ? <FiCheck /> : <FiCopy />}
                        </button>
                      </div>
                    </div>
                    <div className="p-5 lg:p-6 bg-slate-50 rounded-2xl border border-slate-100">
                      <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Content Type</h4>
                      <code className="block w-full px-4 py-3 bg-white text-slate-700 rounded-xl text-xs lg:text-sm font-mono border border-slate-200 shadow-sm">
                        application/json
                      </code>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            {/* Authentication Section */}
            <section id="authentication" className="scroll-mt-32">
              <div className="bg-white rounded-3xl p-6 lg:p-8 shadow-sm border border-slate-100">
                <div className="flex items-center gap-4 mb-6">
                  <div className="p-3 bg-slate-900 text-white rounded-2xl shadow-lg shadow-slate-900/20">
                    <FiServer className="w-6 h-6" />
                  </div>
                  <h2 className="text-2xl lg:text-3xl font-bold text-slate-900">Authentication</h2>
                </div>
                <div className="bg-amber-50 border border-amber-100 p-6 rounded-2xl flex flex-col sm:flex-row gap-4">
                  <div className="text-amber-500 mt-1">
                    <FiServer className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="text-amber-900 font-bold mb-1">Development Mode</h4>
                    <p className="text-amber-800/80 text-sm leading-relaxed">
                      Currently, the API is open for development and testing purposes. 
                      No API key is required for local requests. Future production versions will require Bearer Token authentication.
                    </p>
                  </div>
                </div>
              </div>
            </section>

            <div className="h-px bg-gradient-to-r from-transparent via-slate-200 to-transparent"></div>

            {/* Endpoints List */}
            <div className="space-y-16 lg:space-y-24">
              {endpoints.map((endpoint) => (
                <section key={endpoint.id} id={endpoint.id} className="scroll-mt-32 group">
                  {/* Endpoint Header */}
                  <div className="flex flex-col sm:flex-row items-start gap-4 sm:gap-6 mb-6 lg:mb-8">
                    <div className={`
                      p-4 rounded-2xl bg-${endpoint.color}-50 text-${endpoint.color}-600 
                      ring-1 ring-${endpoint.color}-100 shadow-lg shadow-${endpoint.color}-500/10
                      group-hover:scale-110 transition-transform duration-300
                    `}>
                      <endpoint.icon className="w-6 h-6 lg:w-8 lg:h-8" />
                    </div>
                    <div>
                      <div className="flex flex-wrap items-center gap-3 lg:gap-4 mb-2">
                        <h3 className="text-xl lg:text-2xl font-bold text-slate-900">
                          {endpoint.title}
                        </h3>
                        <MethodBadge method={endpoint.method} />
                      </div>
                      <p className="text-slate-500 text-base lg:text-lg leading-relaxed max-w-2xl">
                        {endpoint.description}
                      </p>
                    </div>
                  </div>

                  {/* Endpoint Card */}
                  <div className="bg-white rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100 overflow-hidden transition-shadow hover:shadow-2xl hover:shadow-slate-200/60">
                    
                    {/* URL Bar */}
                    <div className="flex items-center justify-between px-4 lg:px-6 py-4 bg-slate-50/50 border-b border-slate-100 gap-4">
                      <div className="flex-1 overflow-x-auto custom-scrollbar">
                        <code className="block text-xs lg:text-sm font-mono text-slate-600 px-2 py-1 bg-white rounded-md border border-slate-200 whitespace-nowrap">
                          {endpoint.path}
                        </code>
                      </div>
                      <button 
                        onClick={() => handleCopy(endpoint.path, endpoint.id)}
                        className="p-2 text-slate-400 hover:text-slate-600 hover:bg-white rounded-lg transition-all flex-shrink-0"
                        title="Copy URL"
                      >
                        {copiedId === endpoint.id ? <FiCheck className="w-5 h-5 text-emerald-500" /> : <FiCopy className="w-5 h-5" />}
                      </button>
                    </div>

                    {/* Content Grid */}
                    <div className="grid xl:grid-cols-2 divide-y xl:divide-y-0 xl:divide-x divide-slate-100">
                      
                      {/* Left Column: Parameters */}
                      <div className="p-6 lg:p-8">
                        {endpoint.params.length > 0 ? (
                          <>
                            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-6">Parameters</h4>
                            <div className="space-y-4">
                              {endpoint.params.map((param, idx) => (
                                <div key={idx} className="group/param p-4 rounded-xl bg-slate-50 hover:bg-slate-100 transition-colors border border-slate-100">
                                  <div className="flex flex-wrap items-center justify-between mb-2 gap-2">
                                    <div className="flex items-center gap-2">
                                      <span className="font-mono text-sm font-bold text-slate-700">{param.name}</span>
                                      <span className="text-[10px] lg:text-xs text-slate-400 px-2 py-0.5 bg-white rounded-full border border-slate-200">{param.type}</span>
                                    </div>
                                    {param.required && (
                                      <span className="text-[10px] font-bold text-rose-500 bg-rose-50 px-2 py-1 rounded-full uppercase tracking-wide">Required</span>
                                    )}
                                  </div>
                                  <p className="text-sm text-slate-500 leading-relaxed">{param.desc}</p>
                                </div>
                              ))}
                            </div>
                          </>
                        ) : (
                          <div className="h-full flex flex-col items-center justify-center text-center p-8 text-slate-400">
                            <div className="w-12 h-12 rounded-full bg-slate-50 flex items-center justify-center mb-3">
                              <FiBox className="w-6 h-6 opacity-50" />
                            </div>
                            <p className="text-sm">No parameters required</p>
                          </div>
                        )}
                      </div>

                      {/* Right Column: Code Examples */}
                      <div className="bg-slate-900 p-6 lg:p-8 overflow-hidden relative">
                        <div className="absolute top-0 right-0 p-4">
                          <div className="flex gap-2">
                            <div className="w-3 h-3 rounded-full bg-slate-700"></div>
                            <div className="w-3 h-3 rounded-full bg-slate-700"></div>
                          </div>
                        </div>

                        <div className="space-y-8">
                          {/* Request Body */}
                          {endpoint.body && (
                            <div>
                              <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                                <FiChevronRight className="w-4 h-4 text-orange-500" /> Request Body
                              </h4>
                              <div className="relative group/code">
                                <pre className="text-xs font-mono text-slate-300 overflow-x-auto custom-scrollbar pb-2">
                                  <code>{typeof endpoint.body === 'string' ? endpoint.body : JSON.stringify(endpoint.body, null, 2)}</code>
                                </pre>
                              </div>
                            </div>
                          )}

                          {/* Response Body */}
                          <div>
                            <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                              <FiChevronRight className="w-4 h-4 text-emerald-500" /> Response
                            </h4>
                            <div className="relative group/code">
                              <button 
                                onClick={() => handleCopy(JSON.stringify(endpoint.response, null, 2), `${endpoint.id}-resp`)}
                                className="absolute -top-1 right-0 p-1.5 text-slate-500 hover:text-white bg-white/5 hover:bg-white/10 rounded-lg opacity-0 group-hover/code:opacity-100 transition-all"
                              >
                                {copiedId === `${endpoint.id}-resp` ? <FiCheck className="w-4 h-4" /> : <FiCopy className="w-4 h-4" />}
                              </button>
                              <pre className="text-xs font-mono text-emerald-400 overflow-x-auto custom-scrollbar pb-2">
                                <code>{JSON.stringify(endpoint.response, null, 2)}</code>
                              </pre>
                            </div>
                          </div>
                        </div>
                      </div>

                    </div>
                  </div>
                </section>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default APIPage;
