import { useState } from 'react';

const BreedsPage = () => {
  const [selectedType, setSelectedType] = useState('All');

  const breeds = [
    // Cattle
    { name: 'Gir', type: 'Cattle', region: 'Gujarat', feature: 'High milk yield, heat tolerance' },
    { name: 'Sahiwal', type: 'Cattle', region: 'Punjab', feature: 'Excellent milk producer' },
    { name: 'Red Sindhi', type: 'Cattle', region: 'Sindh/Rajasthan', feature: 'Drought resistant, good milker' },
    { name: 'Tharparkar', type: 'Cattle', region: 'Rajasthan', feature: 'Dual purpose, drought hardy' },
    { name: 'Rathi', type: 'Cattle', region: 'Rajasthan', feature: 'Adaptable to arid regions' },
    { name: 'Hariana', type: 'Cattle', region: 'Haryana', feature: 'Draft and milk production' },
    { name: 'Ongole', type: 'Cattle', region: 'Andhra Pradesh', feature: 'Large body, heat tolerance' },
    { name: 'Kankrej', type: 'Cattle', region: 'Gujarat/Rajasthan', feature: 'Dual purpose breed' },
    { name: 'Deoni', type: 'Cattle', region: 'Maharashtra', feature: 'Good milk and draft' },
    { name: 'Amritmahal', type: 'Cattle', region: 'Karnataka', feature: 'Excellent draught breed' },
    { name: 'Hallikar', type: 'Cattle', region: 'Karnataka', feature: 'Strong draught power' },
    { name: 'Kangayam', type: 'Cattle', region: 'Tamil Nadu', feature: 'Draft and agricultural work' },
    // Buffalo
    { name: 'Murrah', type: 'Buffalo', region: 'Haryana', feature: 'Highest milk yield' },
    { name: 'Mehsana', type: 'Buffalo', region: 'Gujarat', feature: 'High fat content milk' },
    { name: 'Jaffarabadi', type: 'Buffalo', region: 'Gujarat', feature: 'Large size, good milk' },
    { name: 'Surti', type: 'Buffalo', region: 'Gujarat', feature: 'Rich milk quality' },
    { name: 'Nili-Ravi', type: 'Buffalo', region: 'Punjab', feature: 'Excellent milk producer' },
    { name: 'Bhadawari', type: 'Buffalo', region: 'UP/MP', feature: 'Small, hardy breed' }
  ];

  const types = ['All', 'Cattle', 'Buffalo'];
  const filteredBreeds = selectedType === 'All' 
    ? breeds 
    : breeds.filter(b => b.type === selectedType);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-50">
      {/* Hero Section */}
      <section className="py-20 sm:py-24 lg:py-32 bg-white">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center space-y-8">
            <div className="inline-block px-6 py-2 bg-black/5 rounded-full text-sm font-semibold">
              Indigenous Livestock Heritage
            </div>
            <h1 className="text-5xl sm:text-6xl lg:text-7xl xl:text-8xl font-bold text-black tracking-tight">
              Breed Database
            </h1>
            <p className="text-xl sm:text-2xl lg:text-3xl text-gray-600 max-w-4xl mx-auto">
              Explore India's rich diversity of indigenous cattle and buffalo breeds
            </p>
          </div>
        </div>
      </section>

      {/* Filter Tabs */}
      <section className="py-8 bg-white border-b border-gray-200 sticky top-0 z-10 backdrop-blur-sm bg-white/95">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-center gap-4">
            {types.map((type) => (
              <button
                key={type}
                onClick={() => setSelectedType(type)}
                className={`px-8 py-4 rounded-full font-bold transition-all ${
                  selectedType === type
                    ? 'bg-black text-white shadow-xl scale-105'
                    : 'bg-gray-100 text-black hover:bg-gray-200'
                }`}
              >
                {type}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Breeds Grid */}
      <section className="py-16 sm:py-20 lg:py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">
            {filteredBreeds.map((breed, index) => (
              <div
                key={index}
                className="group bg-white rounded-3xl overflow-hidden shadow-lg hover:shadow-2xl transition-all border border-gray-100 hover:border-black"
              >
                {/* Image Placeholder */}
                <div className="relative aspect-[4/3] bg-gradient-to-br from-gray-100 to-gray-200 overflow-hidden">
                  <div className="absolute inset-0 opacity-10">
                    <div className="absolute inset-0" style={{
                      backgroundImage: 'radial-gradient(circle at 2px 2px, black 1px, transparent 0)',
                      backgroundSize: '40px 40px'
                    }}></div>
                  </div>
                  
                  {/* Type Badge */}
                  <div className="absolute top-4 right-4 px-4 py-2 bg-black text-white rounded-full text-sm font-semibold">
                    {breed.type}
                  </div>

                  {/* Breed Name Overlay */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center p-6">
                      <div className="text-4xl sm:text-5xl font-bold text-black/30 group-hover:text-black/50 transition-colors">
                        {breed.name}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Content */}
                <div className="p-6 sm:p-8">
                  <h3 className="text-2xl font-bold text-black mb-3 group-hover:text-gray-700 transition-colors">
                    {breed.name}
                  </h3>
                  
                  <div className="space-y-3">
                    <div className="flex items-center text-sm">
                      <span className="font-semibold text-black mr-2">Region:</span>
                      <span className="text-gray-600">{breed.region}</span>
                    </div>
                    
                    <div className="flex items-start text-sm">
                      <span className="font-semibold text-black mr-2 flex-shrink-0">Features:</span>
                      <span className="text-gray-600">{breed.feature}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Info Section */}
      <section className="py-20 sm:py-24 lg:py-32 bg-black text-white">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center space-y-8">
          <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold">
            Preserving Genetic Heritage
          </h2>
          <p className="text-xl sm:text-2xl text-white/70 max-w-3xl mx-auto leading-relaxed">
            India is home to 43 indigenous cattle breeds and 13 buffalo breeds, 
            each uniquely adapted to local climatic conditions and agricultural practices. 
            Our AI classification system helps document and preserve these invaluable genetic resources.
          </p>
          <div className="pt-6">
            <a href="/classify">
              <button className="px-12 py-6 bg-white text-black font-bold rounded-full hover:bg-white/90 transition-all transform hover:scale-105 shadow-2xl text-lg">
                Classify Your Animal
              </button>
            </a>
          </div>
        </div>
      </section>
    </div>
  );
};

export default BreedsPage;
