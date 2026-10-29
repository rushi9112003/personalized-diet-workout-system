# Personalized Diet and Workout Recommendation System

A complete final-year Machine Learning mini-project built with **Python, Flask, scikit-learn, pandas, NumPy, HTML5, Bootstrap 5, and SQLite**.

The system collects an adult user's physical biometrics, fitness goals, dietary preferences, food allergies, workout experience, and equipment availability to generate:
1. **BMI & Energy Targets** (BMR via Mifflin-St Jeor & TDEE with goal-based caloric & macronutrient distributions).
2. **7-Day Personalized Meal Plan** with exact portions, calories, protein, carbs, fats, and ML similarity match scores.
3. **Weekly Workout Schedule** with targeted muscle groups, exercise instructions, set/rep ranges, rest intervals, and home alternatives.
4. **Local Database Persistence & Text Export** (Save plans to SQLite and download formatted `.txt` summaries with no login required).

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | Python 3.x, Flask (Jinja2 templates) | Web server, routing, session handling, template rendering |
| **Machine Learning** | `scikit-learn` (`NearestNeighbors`, `StandardScaler`) | Content-based similarity ranking via Cosine Distance |
| **Data Processing** | `pandas`, `NumPy` | Dataset ingestion, constraint filtering, feature matrices |
| **Frontend UI** | HTML5, CSS3, Bootstrap 5, FontAwesome 6 | Responsive layout, metrics cards, interactive 7-day tabs |
| **Database** | SQLite3 (`database.db`) | Local persistence of saved recommendation plans |
| **Testing** | `unittest` | Automated unit and integration test suite |

---

## 📁 Project Structure

```
personalized-diet-workout-system/
├── app.py                     # Flask application routes, validation, and export handler
├── ml_engine.py               # ML recommendation engine, biometrics formulas, and split logic
├── database.py                # SQLite database management and CRUD operations
├── requirements.txt           # Python dependencies
├── README.md                  # Complete documentation and viva guide
├── data/
│   ├── foods.csv              # Curated Indian & accessible foods with macros and allergen tags
│   └── workouts.csv           # Exercise repository with muscle groups, equipment, and form cues
├── templates/
│   ├── base.html              # Base template with navbar, footer, modals, and flash alerts
│   ├── index.html             # User profile questionnaire & biometrics input form
│   ├── dashboard.html         # Generated recommendation dashboard (BMI, 7-Day Diet, Workout Split)
│   └── saved_plans.html       # Saved plans listing with view, download, and delete actions
├── static/
│   ├── css/
│   │   └── style.css          # Custom styling and modern card design
│   └── js/
│       └── main.js           # Client-side input validation and loading state feedback
└── tests/
    └── test_system.py         # Unit & integration test suite
```

---

## 🧠 Machine Learning & Recommendation Methodology

### 1. Hard Constraint Filtering (Pre-ML Stage)
Before performing any similarity computations, the engine applies strict logical filtering:
- **Dietary Preference:** Filters candidate foods based on user category (`veg`, `vegan`, `eggetarian`, `non-veg`).
- **Zero-Tolerance Allergen Filter:** Scans the `allergens` field of each food item against the user's selected allergies (e.g., Dairy, Peanuts, Tree Nuts, Gluten, Soy, Eggs, Fish, Shellfish). Any food containing an allergen is completely removed. If an impossible combination leaves 0 candidates, a helpful error message is returned.
- **Workout Equipment & Experience Filter:** Excludes exercises requiring gym equipment if the user only has dumbbells or bodyweight, and matches appropriate difficulty levels.

### 2. Feature Vector Construction & Scaling
- **Food Feature Vector:** Each food candidate is represented as a 5-dimensional numerical vector:
  $$\mathbf{x}_{\text{food}} = [\text{Calories}, \text{Protein (g)}, \text{Carbohydrates (g)}, \text{Fat (g)}, \text{Fiber (g)}]$$
- **Target Meal Vector:** Generated for each meal slot (Breakfast: 25%, Lunch: 35%, Snack: 10%, Dinner: 30% of daily target).
- **Feature Scaling (`StandardScaler`):** Features are standardized to zero mean and unit variance ($\mu=0, \sigma=1$). This prevents features with large absolute numbers (like Calories ~400) from dominating features with smaller scales (like Fiber ~6g).

### 3. Content-Based Similarity Ranking (`NearestNeighbors`)
The system utilizes `sklearn.neighbors.NearestNeighbors` configured with **Cosine Distance**:
$$D_{\text{Cosine}}(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
- Cosine distance measures the angular difference between the user's ideal macronutrient target ratio and the candidate food's macronutrient profile.
- Similarity Match Score is reported as:
  $$\text{Similarity (\%)} = \max(0, 1 - D_{\text{Cosine}}) \times 100$$

### 4. Explicit Rule-Based Post-Processing vs. ML
| Aspect | Method Used | Rationale |
| :--- | :--- | :--- |
| **Nutrient Proportions & Matching** | Machine Learning (Cosine Distance) | Optimal multi-variable ranking across calories, protein, carbs, and fats. |
| **Allergy & Diet Filtering** | Deterministic Rules | Safety-critical; ML ranking must never override medical food allergies. |
| **Portion Size Adjustment** | Rule-Based Scaler ($0.7\times - 1.4\times$) | Aligns individual food portions precisely to target meal calories. |
| **7-Day Schedule Rotation** | Round-Robin Rule Engine | Prevents monotonous meal repetition while maintaining top similarity. |
| **Workout Weekly Split** | Structured Split Logic (3, 4, 5, 6 Days) | Follows exercise physiology principles (Push/Pull/Legs/Upper/Lower/Full Body) with mandatory rest days. |

---

## 📊 Dataset Schema

### `data/foods.csv`
- `food_id` *(int)*: Unique identifier.
- `name` *(string)*: Name of the food item or recipe (e.g., *Paneer Bhurji with 2 Phulkas*, *Dal Tadka with Brown Rice*).
- `meal_type` *(string)*: `breakfast`, `lunch`, `dinner`, or `snack`.
- `category` *(string)*: `Grain`, `Legume`, `Dairy`, `Poultry`, `Seafood`, `Protein`, `Vegetable`.
- `calories` *(float)*: Energy content per default serving (kcal).
- `protein_g`, `carbs_g`, `fat_g`, `fiber_g` *(float)*: Macronutrients and dietary fiber in grams.
- `diet_type` *(string)*: `veg`, `non-veg`, `vegan`, `eggetarian`.
- `allergens` *(string)*: Semicolon-separated allergen labels (e.g., `dairy;gluten`, `peanuts`, `none`).
- `default_serving_unit` *(string)*: Human-readable standard portion size.
- `default_serving_weight_g` *(float)*: Weight in grams for the default serving.

### `data/workouts.csv`
- `exercise_id` *(int)*: Unique identifier.
- `name` *(string)*: Exercise name (e.g., *Goblet Squat*, *Dumbbell Bent-Over Row*).
- `target_muscle` *(string)*: `Chest`, `Back`, `Legs`, `Shoulders`, `Arms`, `Core`, `Full Body`.
- `category` *(string)*: `Push`, `Pull`, `Legs`, `Core`, `Cardio`, `Full Body`.
- `difficulty_level` *(string)*: `beginner`, `intermediate`, `advanced`.
- `equipment_required` *(string)*: `bodyweight`, `dumbbells`, `gym`.
- `intensity_score` *(float)*: Relative intensity rating (1.0 to 10.0 scale).
- `calorie_burn_per_min` *(float)*: Estimated metabolic burn rate per minute.
- `default_sets` *(int)*: Recommended working sets.
- `default_reps` *(string)*: Target repetition range.
- `rest_seconds` *(int)*: Inter-set rest period in seconds.
- `instructions` *(string)*: Step-by-step biomechanical execution steps.
- `form_cues` *(string)*: Critical posture and safety cue.
- `home_alternative` *(string)*: Zero-equipment home substitute exercise.

---

## 🚀 Quickstart & Installation

### Step 1: Clone or Navigate to the Project Directory
```bash
cd personalized-diet-workout-system
```

### Step 2: Create and Activate a Python Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Automated Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Step 5: Start the Flask Application
```bash
python app.py
```
Open your web browser and navigate to: **`http://127.0.0.1:5000`**

---

## 🧪 Running Tests

The test suite validates biometric calculations, allergy exclusion integrity, ML recommendation structure, database persistence, and web routes:
```bash
python tests/test_system.py
```

Expected output:
```text
......
----------------------------------------------------------------------
Ran 8 tests in 0.450s

OK
```

---

## 🎓 Final-Year Project Viva / Interview Questions & Answers

### Q1: What type of Recommendation System is implemented in this project?
> **Answer:** This project implements a **Content-Based Recommendation System**. Instead of relying on user-item interaction histories (Collaborative Filtering), it matches the detailed attribute profile of items (such as calories, protein, carbohydrates, fats, and fiber in foods; intensity and calorie burn in workouts) with the user's specific target requirements using cosine distance in a standardized feature space.

### Q2: Why is Content-Based Filtering preferred over Collaborative Filtering for this project?
> **Answer:** 
> 1. **Cold Start Problem:** In fitness and diet planning, a new user provides biometrics and needs an immediate, mathematically sound recommendation without waiting for thousands of other users to rate meals.
> 2. **Strict Health & Allergy Constraints:** Collaborative filtering cannot guarantee zero-tolerance allergen safety because it recommends items based on similarity among users. Content-based filtering allows deterministic pre-filtering for allergies and diet types before ranking.
> 3. **Privacy & Simplicity:** It operates locally without requiring user tracking, ratings, or external data pipelines.

### Q3: Why did you use Cosine Distance instead of Euclidean Distance in `NearestNeighbors`?
> **Answer:** Euclidean distance measures absolute difference in magnitude, whereas **Cosine Distance** measures the angle between vectors (i.e., directional similarity / proportions). In nutrition, the *ratio* of macronutrients (e.g., 30% protein, 45% carbs, 25% fat) is the primary indicator of diet quality. Cosine distance identifies foods with the exact macro ratio balance desired, and rule-based portion scaling subsequently adjusts the absolute quantity.

### Q4: Why is `StandardScaler` necessary before applying NearestNeighbors?
> **Answer:** The features in our dataset operate on drastically different numerical scales. For example, Calories range between 100–500 kcal, whereas Fiber ranges between 1–10 grams. Without standardization, the distance calculation would be almost entirely dominated by calories, ignoring protein and fiber. `StandardScaler` standardizes each feature to have a mean of 0 and standard deviation of 1 ($\mu=0, \sigma=1$), ensuring equal importance across all nutritional dimensions.

### Q5: How are scientific formulas like BMR and TDEE integrated with ML?
> **Answer:** 
> - **BMR (Basal Metabolic Rate):** Calculated using the clinically validated **Mifflin-St Jeor equation**, accounting for biological gender, age, weight, and height.
> - **TDEE (Total Daily Energy Expenditure):** Calculated by multiplying BMR with the user's physical activity factor (1.2 to 1.9).
> - Target calories and macronutrients are derived based on the user's goal (e.g., 450 kcal deficit for weight loss, 350 kcal surplus for muscle gain).
> - These calculated targets serve as the **query vectors** for the NearestNeighbors ML algorithm.

### Q6: How does the system ensure safety regarding food allergies?
> **Answer:** Allergy safety is treated as a non-negotiable hard constraint. It is executed in the pre-filtering pipeline before the ML model is invoked. If an item contains any allergen specified by the user (such as peanuts, dairy, gluten, or shellfish), it is permanently removed from the candidate pool. If an overly restrictive combination leaves no options, the system gracefully halts and notifies the user with actionable feedback rather than making dangerous assumptions.

### Q7: What are the limitations of this project and scope for future work?
> **Answer:**
> 1. **Dataset Scope:** The current dataset uses curated common Indian and accessible foods; expanding to larger open food databases (e.g., USDA / Open Food Facts) would increase variety.
> 2. **Static vs. Adaptive Feedback:** Currently, the system generates static 7-day plans. Future iterations can incorporate reinforcement learning or user feedback loops to adjust recommendations based on weekly bodyweight changes.
> 3. **Medical Disclaimer:** The system is an educational adult fitness tool and does not replace medical supervision for clinical conditions (e.g., diabetes, renal disease).

---

## 📜 Educational Disclaimer
This software is developed strictly as an educational mini-project for academic demonstrations. The nutritional estimates and exercise schedules are generated based on generalized mathematical formulas and sample datasets. It does not constitute certified medical, clinical, or dietary advice.
