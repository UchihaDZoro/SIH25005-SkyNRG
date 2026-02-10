"""
Official Type Classification Traits as per Annex II
Following exact government format with descriptors and measurements
"""

TRAIT_DEFINITIONS = {
    "Strength": [
        {
            "name": "Stature",
            "low_descriptor": "Short",
            "high_descriptor": "Tall",
            "measurement_unit": "cm",
            "description": "Overall height and frame size"
        },
        {
            "name": "Heart Girth",
            "low_descriptor": "Narrow",
            "high_descriptor": "Wide",
            "measurement_unit": "cm",
            "description": "Chest depth and width capacity"
        },
        {
            "name": "Body Length",
            "low_descriptor": "Short",
            "high_descriptor": "Long",
            "measurement_unit": "cm",
            "description": "Distance from shoulder to pin bone"
        },
        {
            "name": "Body Depth",
            "low_descriptor": "Shallow",
            "high_descriptor": "Deep",
            "measurement_unit": "cm",
            "description": "Rib depth and barrel capacity"
        },
        {
            "name": "Angularity",
            "low_descriptor": "Non-angular",
            "high_descriptor": "Angular",
            "measurement_unit": None,
            "description": "Overall dairy character and sharpness"
        }
    ],
    "Rump": [
        {
            "name": "Rump Angle",
            "low_descriptor": "High",
            "high_descriptor": "Low",
            "measurement_unit": "degrees",
            "description": "Slope from hips to pin bones"
        },
        {
            "name": "Rump Width",
            "low_descriptor": "Narrow",
            "high_descriptor": "Wide",
            "measurement_unit": "cm",
            "description": "Distance between pin bones"
        }
    ],
    "Feet and Leg": [
        {
            "name": "Rear Legs Set",
            "low_descriptor": "Straight",
            "high_descriptor": "Curved",
            "measurement_unit": None,
            "description": "Angle of rear legs from side view"
        },
        {
            "name": "Rear Legs Rear View",
            "low_descriptor": "Hocked-in",
            "high_descriptor": "Straight",
            "measurement_unit": None,
            "description": "Straightness from rear view"
        },
        {
            "name": "Foot Angle",
            "low_descriptor": "Low",
            "high_descriptor": "Steep",
            "measurement_unit": "degrees",
            "description": "Hoof slope angle"
        }
    ],
    "Udder": [
        {
            "name": "Fore Udder Attachment",
            "low_descriptor": "Weak",
            "high_descriptor": "Strong",
            "measurement_unit": None,
            "description": "Strength of front attachment to body wall"
        },
        {
            "name": "Rear Udder Height",
            "low_descriptor": "Low",
            "high_descriptor": "High",
            "measurement_unit": "cm",
            "description": "Height of rear attachment above hock"
        },
        {
            "name": "Central Ligament",
            "low_descriptor": "Weak",
            "high_descriptor": "Strong",
            "measurement_unit": None,
            "description": "Strength of median suspensory ligament"
        },
        {
            "name": "Udder Depth",
            "low_descriptor": "Deep",
            "high_descriptor": "Shallow",
            "measurement_unit": "cm",
            "description": "Distance from hock to udder floor"
        },
        {
            "name": "Front Teat Placement",
            "low_descriptor": "Wide",
            "high_descriptor": "Close",
            "measurement_unit": "cm",
            "description": "Distance between front teats"
        },
        {
            "name": "Teat Length",
            "low_descriptor": "Short",
            "high_descriptor": "Long",
            "measurement_unit": "cm",
            "description": "Length of teats"
        },
        {
            "name": "Rear Teat Placement",
            "low_descriptor": "Wide",
            "high_descriptor": "Close",
            "measurement_unit": "cm",
            "description": "Distance between rear teats"
        },
        {
            "name": "Rear udder width",
            "low_descriptor": "Narrow",
            "high_descriptor": "Wide",
            "measurement_unit": "cm",
            "description": "Width at rear attachment"
        },
        {
            "name": "Teat thickness",
            "low_descriptor": "Thin",
            "high_descriptor": "Thick",
            "measurement_unit": "cm",
            "description": "Diameter of teats"
        }
    ],
    "General": [
        {
            "name": "Body condition score",
            "low_descriptor": "Thin",
            "high_descriptor": "Fatty",
            "measurement_unit": None,
            "description": "Overall body fat coverage (1-5 BCS scale)"
        }
    ]
}

def get_all_traits_flat():
    """Return flat list of all 20 traits"""
    traits = []
    for category, category_traits in TRAIT_DEFINITIONS.items():
        for trait in category_traits:
            traits.append({
                "category": category,
                **trait
            })
    return traits
