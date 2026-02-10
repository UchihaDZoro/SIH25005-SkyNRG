const BreedsSection = () => {
  const cattleBreeds = [
    { name: 'Gir', origin: 'Gujarat', milk: '2,500 kg/lactation', image: '🐄' },
    { name: 'Sahiwal', origin: 'Punjab', milk: '2,000 kg/lactation', image: '🐄' },
    { name: 'Red Sindhi', origin: 'Sindh', milk: '1,800 kg/lactation', image: '🐄' },
    { name: 'Tharparkar', origin: 'Rajasthan', milk: '1,500 kg/lactation', image: '🐄' }
  ];

  const buffaloBreeds = [
    { name: 'Murrah', origin: 'Haryana', milk: '2,800 kg/lactation', image: '🐃' },
    { name: 'Surti', origin: 'Gujarat', milk: '2,400 kg/lactation', image: '🐃' },
    { name: 'Jaffarabadi', origin: 'Gujarat', milk: '2,200 kg/lactation', image: '🐃' },
    { name: 'Mehsana', origin: 'Gujarat', milk: '2,000 kg/lactation', image: '🐃' }
  ];

  return (
    <section className="py-20 lg:py-28 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16 space-y-4">
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-black">
            Supporting Indigenous Excellence
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Comprehensive classification for India's premium dairy breeds
          </p>
        </div>

        {/* Cattle Breeds */}
        <div className="mb-16">
          <h3 className="text-2xl font-bold text-black mb-6 flex items-center">
            <span className="text-3xl mr-3">🐄</span>
            Indigenous Cattle Breeds
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {cattleBreeds.map((breed, idx) => (
              <div key={idx} className="text-center group bg-white rounded-2xl p-6 border-2 border-gray-200 hover:border-orange-500 hover:bg-orange-50 hover:shadow-xl transition-all">
                <div className="text-6xl mb-4 group-hover:scale-110 transition-transform">
                  {breed.image}
                </div>
                <h4 className="text-xl font-bold text-black mb-2">{breed.name}</h4>
                <div className="space-y-1 text-sm text-gray-600">
                  <p><span className="font-medium">Origin:</span> {breed.origin}</p>
                  <p><span className="font-medium">Milk:</span> {breed.milk}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Buffalo Breeds */}
        <div>
          <h3 className="text-2xl font-bold text-black mb-6 flex items-center">
            <span className="text-3xl mr-3">🐃</span>
            Indigenous Buffalo Breeds
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {buffaloBreeds.map((breed, idx) => (
              <div key={idx} className="text-center group bg-white rounded-2xl p-6 border-2 border-gray-200 hover:border-blue-500 hover:bg-blue-50 hover:shadow-xl transition-all">
                <div className="text-6xl mb-4 group-hover:scale-110 transition-transform">
                  {breed.image}
                </div>
                <h4 className="text-xl font-bold text-black mb-2">{breed.name}</h4>
                <div className="space-y-1 text-sm text-gray-600">
                  <p><span className="font-medium">Origin:</span> {breed.origin}</p>
                  <p><span className="font-medium">Milk:</span> {breed.milk}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default BreedsSection;
