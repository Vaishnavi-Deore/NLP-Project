"""
Advisory Module for AQI-Sense
Provides educational guidance, category descriptions, action recommendations,
AQI category metrics, and essential medical disclaimers.
"""

MEDICAL_DISCLAIMER = (
    "This application provides general educational information and intent classification. "
    "It is not a medical diagnosis or substitute for professional medical advice, clinical diagnosis, "
    "or treatment. If you experience persistent symptoms, severe chest tightness, or respiratory distress, "
    "please consult a qualified healthcare provider immediately."
)

# Advisory information for all 9 classification categories
CATEGORY_ADVISORY = {
    "AQI Information": {
        "title": "Air Quality Index (AQI) Fundamentals",
        "badge_color": "#3B82F6",
        "description": (
            "The Air Quality Index (AQI) is a standardized numerical scale (typically 0 to 500) used by "
            "environmental agencies worldwide to report daily air quality and its associated public health risk."
        ),
        "key_takeaways": [
            "Scale ranges from 0 (pristine air) to 500 (hazardous pollution).",
            "Calculated based on major ambient pollutants: PM2.5, PM10, Ground-level Ozone (O3), Nitrogen Dioxide (NO2), Sulfur Dioxide (SO2), and Carbon Monoxide (CO).",
            "Values below 50 indicate good, healthy air; values above 150 require active exposure reduction.",
            "PM2.5 particles (diameter <= 2.5 micrometers) are considered the most hazardous because they can penetrate deep into lung alveoli and systemic blood circulation."
        ],
        "suggested_actions": [
            "Check local municipal AQI stations during early mornings and late evenings.",
            "Compare AQI readings before planning long outdoor events or school athletics.",
            "Familiarize yourself with official regional color bands (Green, Yellow, Orange, Red, Purple, Maroon)."
        ],
        "dos_and_donts": {
            "dos": ["Monitor daily air quality updates", "Understand how weather and wind speed affect AQI", "Use certified air monitoring apps"],
            "donts": ["Do not assume clear skies guarantee zero invisible ozone or fine particulate pollution", "Do not ignore sustained high AQI levels"]
        }
    },
    "Health Effects": {
        "title": "General Health Effects of Poor Air Quality",
        "badge_color": "#EF4444",
        "description": (
            "Air pollution exerts both acute (immediate) and chronic (long-term cumulative) impacts on the human body, "
            "primarily affecting the cardiovascular and pulmonary systems through oxidative stress and systemic inflammation."
        ),
        "key_takeaways": [
            "Fine particulate matter (PM2.5) induces endothelial dysfunction, arterial stiffness, and microvascular stress.",
            "Prolonged exposure is epidemiologically correlated with elevated risks of ischemic heart disease, stroke, and reduced lung capacity.",
            "Ultrafine particles can translocate across the alveolar-capillary barrier into the bloodstream.",
            "Ground-level ozone acts as a powerful oxidizer, inflaming mucous membranes and triggering bronchial hyperreactivity."
        ],
        "suggested_actions": [
            "Maintain overall cardiovascular health and balanced hydration during high pollution episodes.",
            "Minimize vigorous exertion near high-density traffic arteries where diesel exhaust concentrations are highest.",
            "Consult with your doctor if you have known cardiovascular risk factors."
        ],
        "dos_and_donts": {
            "dos": ["Prioritize antioxidant-rich nutrition", "Reduce time spent near busy freight corridors", "Keep emergency contacts accessible"],
            "donts": ["Do not dismiss prolonged chest pain or palpitations as mere smog discomfort; seek medical care", "Do not smoke or use wood stoves during smog alerts"]
        }
    },
    "Symptoms": {
        "title": "Air Pollution-Related Symptom Awareness",
        "badge_color": "#F59E0B",
        "description": (
            "Inhaling contaminated air can provoke immediate physiological irritation of the mucous membranes, upper respiratory tract, "
            "and ocular surfaces. Knowing common symptoms helps differentiate transient smog irritation from severe conditions."
        ),
        "key_takeaways": [
            "Common upper tract symptoms: Stinging or watery eyes, burning throat, dry hacking cough, post-nasal drip, and sinus congestion.",
            "Systemic symptoms: Smog-induced tension headaches, mild fatigue, and transient dizziness.",
            "Warning signs requiring emergency attention: Severe shortness of breath, cyanosis (bluish lips/nails), intractable wheezing, or crushing chest pain.",
            "Contact lens wearers often report corneal irritation due to micro-particles becoming trapped under lenses."
        ],
        "suggested_actions": [
            "Rinse eyes and face with sterile saline or clean lukewarm water after returning indoors.",
            "Stay well-hydrated to help maintain the protective mucous lining of the respiratory tract.",
            "Switch to prescription eyeglasses instead of contact lenses on high AQI days."
        ],
        "dos_and_donts": {
            "dos": ["Use lubricating eye drops if advised by a pharmacist", "Practice warm steam inhalation for soothing airways", "Rest in a clean indoor room"],
            "donts": ["Do not vigorously rub irritated eyes (can cause corneal micro-abrasions)", "Do not ignore persistent or worsening breathing difficulties"]
        }
    },
    "Precautions": {
        "title": "Indoor & Personal Protective Precautions",
        "badge_color": "#10B981",
        "description": (
            "Proactive behavioral adjustments and indoor air mitigation can substantially reduce personal particulate inhalation dosage "
            "during moderate, severe, and hazardous air quality episodes."
        ),
        "key_takeaways": [
            "Indoor air quality can often deteriorate if outdoor smoke infiltrates and indoor pollution sources (cooking, candles) are present.",
            "True HEPA (High-Efficiency Particulate Air) filtration captures 99.97% of airborne particles down to 0.3 microns.",
            "Recirculation mode in automotive HVAC systems prevents roadside vehicle exhaust from entering the cabin.",
            "Wet-mopping floors rather than dry-sweeping prevents resuspending settled particulate dust."
        ],
        "suggested_actions": [
            "Keep residential windows and doors closed when outdoor AQI is elevated.",
            "Operate a properly sized HEPA air purifier in frequently occupied rooms (e.g., bedrooms).",
            "Use kitchen exhaust hoods while cooking, and avoid burning incense, candles, or wood fires indoors.",
            "Set car climate control to recirculate indoor air when caught in traffic jams."
        ],
        "dos_and_donts": {
            "dos": ["Change or vacuum air purifier filters according to manufacturer specifications", "Ventilate briefly only during periods of lowest outdoor pollution", "Keep a clean-air sanctuary room"],
            "donts": ["Do not dry-sweep dusty rooms without wetting surfaces first", "Do not rely on ozone-generating 'air cleaners' which worsen lung irritation"]
        }
    },
    "Masks and Protection": {
        "title": "Respirator and Mask Guidance",
        "badge_color": "#8B5CF6",
        "description": (
            "Personal respiratory protective equipment provides a physical barrier against airborne particulates. "
            "Filtration efficacy depends strictly on filter certification and achieving a continuous airtight facial seal."
        ),
        "key_takeaways": [
            "Certified particulate respirators (N95, KN95, KF94, FFP2) filter at least 94% to 95% of fine airborne particles.",
            "Standard surgical masks and cloth bandanas leak significantly around the edges and do NOT offer reliable protection against sub-micron PM2.5.",
            "Respirators must create an airtight seal over the bridge of the nose and under the chin; facial hair interferes with proper sealing.",
            "Standard particulate respirators filter solid and liquid aerosols, but do not filter chemical gases unless equipped with activated carbon layers."
        ],
        "suggested_actions": [
            "Choose a certified N95, KN95, or KF94 mask that fits comfortably without gaps around the nose and cheeks.",
            "Perform a user seal check (inhale and exhale sharply while checking for perimeter air leaks) before entering heavy pollution.",
            "Replace disposable respirators when breathing resistance increases or if the mask becomes soiled, moist, or deformed."
        ],
        "dos_and_donts": {
            "dos": ["Adjust the moldable metal nose-clip firmly across the bridge of your nose", "Store clean respirators in breathable paper bags", "Opt for dual head-strap models for tighter seal"],
            "donts": ["Do not wash disposable N95 respirators with water or alcohol (destroys electrostatic charge)", "Do not wear loose surgical masks expecting PM2.5 protection"]
        }
    },
    "Outdoor Activities": {
        "title": "Outdoor Exercise & Activity Guidelines",
        "badge_color": "#EC4899",
        "description": (
            "During physical exertion, minute ventilation (the volume of air inhaled per minute) increases by up to 10 to 20 times. "
            "Heavy mouth-breathing bypasses natural nasal filtration, drastically increasing internal pollutant deposition."
        ),
        "key_takeaways": [
            "When AQI is Moderate (51-100): Outdoor exercise is generally safe for healthy individuals; sensitive persons should monitor symptoms.",
            "When AQI is Unhealthy for Sensitive Groups (101-150): Children, seniors, and respiratory patients should reduce prolonged outdoor cardio.",
            "When AQI is Unhealthy (151-200): Everyone should reduce vigorous outdoor workouts; shift high-intensity cardio indoors.",
            "When AQI is Very Unhealthy or Hazardous (>200): Avoid all strenuous outdoor sports, running, and heavy manual labor."
        ],
        "suggested_actions": [
            "Substitute outdoor jogging or cycling with indoor treadmill running, yoga, or indoor gymnasium workouts.",
            "Schedule unavoidable outdoor walks during midday when morning inversions have dispersed and before evening traffic peaks.",
            "Reduce workout duration and intensity if outdoor exercise cannot be relocated."
        ],
        "dos_and_donts": {
            "dos": ["Check hourly AQI forecasts before morning or evening runs", "Move workouts into air-filtered gym facilities", "Listen to your body and stop if chest tightness occurs"],
            "donts": ["Do not sprint or do interval cardio next to busy highways", "Do not force young children to participate in prolonged outdoor athletic meets during air alerts"]
        }
    },
    "Vulnerable Groups": {
        "title": "Protection for Sensitive and High-Risk Demographics",
        "badge_color": "#06B6D4",
        "description": (
            "Certain biological and physiological factors render specific populations significantly more susceptible "
            "to adverse consequences from particulate matter and air contaminants."
        ),
        "key_takeaways": [
            "Infants & Children: Their lungs are still developing, they breathe more air per pound of body weight, and spend more time playing outdoors.",
            "Senior Citizens (65+): Age-associated decline in physiological resilience and higher prevalence of subclinical cardiovascular disease increase vulnerability.",
            "Pregnant Women: Particulate exposure has been associated with pregnancy complications, fetal oxidative stress, and intrauterine growth restriction.",
            "Pre-existing Conditions: People with cardiovascular disease, diabetes, hypertension, and immune deficiencies require strict exposure mitigation."
        ],
        "suggested_actions": [
            "Ensure vulnerable family members remain inside well-ventilated, filtered indoor spaces during air pollution alerts.",
            "Schools and childcare facilities should enforce indoor recess policies when the AQI exceeds 100-150.",
            "Keep baseline vital signs and essential medications up to date for seniors and chronically ill dependents."
        ],
        "dos_and_donts": {
            "dos": ["Consult pediatricians and obstetricians for personalized air protection plans", "Keep senior living spaces equipped with certified HEPA filtration", "Monitor high-risk individuals closely"],
            "donts": ["Do not expose newborns or toddlers to outdoor morning smog in open strollers", "Do not delay seeking clinical assistance if vulnerable individuals display altered breathing"]
        }
    },
    "Respiratory Health": {
        "title": "Air Quality and Chronic Pulmonary Conditions",
        "badge_color": "#F97316",
        "description": (
            "Air pollution directly irritates the tracheobronchial tree and can provoke rapid acute exacerbations "
            "in patients living with Asthma, Chronic Obstructive Pulmonary Disease (COPD), Bronchitis, and Bronchiectasis."
        ),
        "key_takeaways": [
            "PM2.5 and Ozone trigger airway hyperreactivity, bronchial smooth muscle spasms, and excessive mucus hypersecretion.",
            "Asthma attacks can occur even with brief spikes in ambient particulate concentration.",
            "Exposure to fine particulate matter impairs alveolar macrophage defense, raising susceptibility to secondary viral and bacterial chest infections.",
            "Regular adherence to prescribed controller and preventer inhalers is vital during pollution-prone seasons."
        ],
        "suggested_actions": [
            "Always keep fast-acting bronchodilator rescue inhalers readily accessible at home, work, and during transit.",
            "Review your personalized Asthma Action Plan or COPD management protocol with your pulmonologist prior to winter smog spikes.",
            "Monitor peak expiratory flow (PEF) readings daily to catch early signs of airway narrowing before severe symptoms emerge."
        ],
        "dos_and_donts": {
            "dos": ["Use a spacer device with metered-dose inhalers for maximum medication lung deposition", "Rinse mouth after steroid inhaler use", "Keep spare rescue medications on hand"],
            "donts": ["Never discontinue prescribed controller inhalers without consulting your physician", "Do not ignore nocturnal wheezing or coughing that wakes you from sleep"]
        }
    },
    "Pollution Exposure": {
        "title": "Dosage, Duration, and Cumulative Exposure Risks",
        "badge_color": "#6366F1",
        "description": (
            "The physiological impact of air pollution depends on cumulative dose—a product of ambient concentration, "
            "exposure duration, and individual minute ventilation. Managing cumulative exposure is key to mitigating lifelong risk."
        ),
        "key_takeaways": [
            "Acute Exposure: Brief spikes (hours to days) can provoke immediate mucous irritation, cardiac stress, and asthma exacerbations.",
            "Chronic Exposure: Cumulative daily exposure over months and years contributes to vascular remodeling, pulmonary fibrosis, and shortened life expectancy.",
            "Occupational Vulnerability: Outdoor delivery drivers, traffic police, and construction workers endure significantly higher cumulative doses.",
            "Living within 300 to 500 meters of high-density highways correlates with markedly higher lifetime particulate burdens."
        ],
        "suggested_actions": [
            "Implement exposure reduction strategies: plan commute routes away from heavy freight corridors and avoid peak congestion periods.",
            "Outdoor occupational workers should insist on certified personal protective respirators and rotate shifts away from pollution hotspots.",
            "Utilize indoor air purifiers to ensure that night-time sleeping hours (approx. 8 hours) provide an uncompromised clean-air recovery window."
        ],
        "dos_and_donts": {
            "dos": ["Calculate personal daily exposure windows and minimize unnecessary outdoor linger time", "Advocate for municipal emissions controls and dust mitigation", "Ensure recovery periods in clean indoor air"],
            "donts": ["Do not assume that brief daily exposure has zero effect if outdoor levels are hazardous", "Do not perform strenuous physical labor without certified respiratory defense"]
        }
    }
}

# EPA AQI Breakdown Table
AQI_LEVELS = [
    {
        "range": (0, 50),
        "category": "Good",
        "color": "#10B981", # Green
        "description": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
        "general_guidance": "Enjoy normal outdoor activities.",
        "sensitive_guidance": "None. Ideal air quality for everyone.",
        "outdoor_activity": "Ideal conditions for all outdoor sports, running, and recreation.",
        "mask_recommended": False
    },
    {
        "range": (51, 100),
        "category": "Moderate",
        "color": "#FBBF24", # Yellow
        "description": "Air quality is acceptable; however, there may be a moderate health concern for a very small number of unusually sensitive individuals.",
        "general_guidance": "Air quality is generally acceptable for most people.",
        "sensitive_guidance": "Unusually sensitive people with chronic respiratory conditions should consider reducing prolonged or heavy outdoor exertion.",
        "outdoor_activity": "Safe for regular outdoor activities. Sensitive individuals should observe how they feel.",
        "mask_recommended": False
    },
    {
        "range": (101, 150),
        "category": "Unhealthy for Sensitive Groups",
        "color": "#F97316", # Orange
        "description": "Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
        "general_guidance": "General public can still engage in outdoor activities, but take periodic rest breaks.",
        "sensitive_guidance": "Children, older adults, and people with heart or lung disease should reduce prolonged outdoor exertion.",
        "outdoor_activity": "Cut back on vigorous cardio sessions outside; shift intense workouts indoors.",
        "mask_recommended": True
    },
    {
        "range": (151, 200),
        "category": "Unhealthy",
        "color": "#EF4444", # Red
        "description": "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.",
        "general_guidance": "Everyone should reduce prolonged or heavy exertion outdoors. Keep residential windows closed.",
        "sensitive_guidance": "Sensitive groups must avoid prolonged outdoor exertion; stay in air-filtered indoor spaces.",
        "outdoor_activity": "Avoid outdoor running, sports practices, and strenuous manual labor. Move exercise indoors.",
        "mask_recommended": True
    },
    {
        "range": (201, 300),
        "category": "Very Unhealthy",
        "color": "#8B5CF6", # Purple
        "description": "Health alert: The risk of health effects is increased for everyone.",
        "general_guidance": "Active children and adults, and people with respiratory disease, should avoid all outdoor exertion; everyone else should severely limit outdoor exertion.",
        "sensitive_guidance": "Sensitive groups should remain indoors and keep physical activity levels minimal.",
        "outdoor_activity": "All outdoor sporting events, school recess, and workouts should be cancelled or moved indoors.",
        "mask_recommended": True
    },
    {
        "range": (301, 500),
        "category": "Hazardous",
        "color": "#881337", # Maroon
        "description": "Health warning of emergency conditions: The entire population is more likely to be affected.",
        "general_guidance": "Everyone should avoid all physical activity outdoors. Stay inside with high-efficiency air cleaners running.",
        "sensitive_guidance": "Emergency health conditions. High-risk groups should remain in sealed clean rooms.",
        "outdoor_activity": "Strict prohibition on outdoor physical exertion. Essential outdoor travel requires certified N95+ respirators.",
        "mask_recommended": True
    }
]

def get_aqi_details(aqi_val: float) -> dict:
    """
    Returns AQI category details for a given AQI number.
    """
    for level in AQI_LEVELS:
        low, high = level["range"]
        if low <= aqi_val <= high or (high == 500 and aqi_val > 500):
            return level
    # Default to hazardous if > 500
    if aqi_val > 500:
        return AQI_LEVELS[-1]
    return AQI_LEVELS[0]

def get_category_advisory(category_name: str) -> dict:
    """
    Returns advisory payload for a predicted category.
    """
    return CATEGORY_ADVISORY.get(category_name, {
        "title": "Air Quality Guidance",
        "badge_color": "#6B7280",
        "description": "General air quality and health guidance.",
        "key_takeaways": ["Stay informed on local air quality metrics.", "Protect sensitive individuals."],
        "suggested_actions": ["Check AQI forecasts before planning activities."],
        "dos_and_donts": {
            "dos": ["Follow municipal recommendations"],
            "donts": ["Do not ignore elevated pollution alerts"]
        }
    })
