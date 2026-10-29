"""
ml_engine.py
============
Machine Learning and Rule-Based Recommendation Engine for:
1. Nutritional calculations (BMI, BMR, TDEE, Macros)
2. Content-Based Food Recommendation (scikit-learn NearestNeighbors with Cosine Distance)
3. Content-Based Workout Recommendation (scikit-learn NearestNeighbors with Cosine Distance)
4. Explicit rule-based portion scaling and weekly schedule synthesis.

Educational Note:
All nutrition estimates and workout templates are for adult educational demonstration
and not medical or clinical prescriptions.
"""

import os
import pandas as pd
import numpy as np
try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
except Exception:
    # NumPy fallback implementation with exact scikit-learn API & cosine distance
    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None

        def fit(self, X):
            X = np.asarray(X, dtype=float)
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            self.scale_[self.scale_ == 0.0] = 1.0
            return self

        def transform(self, X):
            X = np.asarray(X, dtype=float)
            return (X - self.mean_) / self.scale_

        def fit_transform(self, X):
            return self.fit(X).transform(X)

    class NearestNeighbors:
        def __init__(self, n_neighbors=5, metric='cosine', algorithm='brute'):
            self.n_neighbors = n_neighbors
            self.metric = metric
            self.algorithm = algorithm
            self.X_ = None

        def fit(self, X):
            self.X_ = np.asarray(X, dtype=float)
            return self

        def kneighbors(self, target_vector):
            target = np.asarray(target_vector, dtype=float)
            if target.ndim == 1:
                target = target.reshape(1, -1)
            
            # Cosine distance: 1 - (u . v) / (||u|| * ||v||)
            dot_products = np.dot(self.X_, target.T).flatten()
            norm_X = np.linalg.norm(self.X_, axis=1)
            norm_target = np.linalg.norm(target)
            
            # Prevent division by zero
            denom = norm_X * norm_target
            denom[denom == 0.0] = 1e-9
            
            cosine_similarity = dot_products / denom
            cosine_distances = 1.0 - cosine_similarity
            
            # Get top k closest (smallest cosine distance)
            sorted_indices = np.argsort(cosine_distances)
            k = min(self.n_neighbors, len(sorted_indices))
            top_k_indices = sorted_indices[:k]
            top_k_distances = cosine_distances[top_k_indices]
            
            return np.array([top_k_distances]), np.array([top_k_indices])

# Base path for datasets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FOODS_CSV = os.path.join(DATA_DIR, 'foods.csv')
WORKOUTS_CSV = os.path.join(DATA_DIR, 'workouts.csv')


def load_datasets():
    """Load local CSV datasets for foods and workouts."""
    foods_df = pd.read_csv(FOODS_CSV)
    workouts_df = pd.read_csv(WORKOUTS_CSV)
    return foods_df, workouts_df


# -------------------------------------------------------------------------
# 1. BIOMETRICS & NUTRITION CALCULATORS (Rule-Based Scientific Formulas)
# -------------------------------------------------------------------------

def calculate_bmi(weight_kg, height_cm):
    """
    Calculate Body Mass Index (BMI) and categorize.
    Formula: weight (kg) / (height (m))^2
    """
    height_m = height_cm / 100.0
    bmi = round(weight_kg / (height_m ** 2), 1)
    
    if bmi < 18.5:
        category = "Underweight"
        explanation = "Your BMI indicates you are underweight. A nutrient-dense surplus is recommended."
        badge_class = "warning"
    elif 18.5 <= bmi <= 24.9:
        category = "Normal weight"
        explanation = "Your BMI is in the healthy range. Focus on balanced nutrition and fitness performance."
        badge_class = "success"
    elif 25.0 <= bmi <= 29.9:
        category = "Overweight"
        explanation = "Your BMI indicates slight overweight. A moderate calorie deficit with resistance training is effective."
        badge_class = "warning"
    else:
        category = "Obese"
        explanation = "Your BMI is in the obese range. A structured, sustainable calorie deficit and daily physical activity is recommended."
        badge_class = "danger"

    return {
        "bmi": bmi,
        "category": category,
        "explanation": explanation,
        "badge_class": badge_class
    }


def calculate_bmr(weight_kg, height_cm, age, gender):
    """
    Calculate Basal Metabolic Rate (BMR) using the Mifflin-St Jeor Equation.
    Men: (10 × weight) + (6.25 × height) - (5 × age) + 5
    Women: (10 × weight) + (6.25 × height) - (5 × age) - 161
    Other: Average of both
    """
    base = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    if gender.lower() == 'male':
        bmr = base + 5
    elif gender.lower() == 'female':
        bmr = base - 161
    else:
        bmr = base - 78
    return round(bmr)


def calculate_tdee_and_targets(weight_kg, height_cm, age, gender, activity_level, goal):
    """
    Calculate Total Daily Energy Expenditure (TDEE) and target macronutrients.
    """
    bmr = calculate_bmr(weight_kg, height_cm, age, gender)
    
    activity_multipliers = {
        'sedentary': 1.2,        # Little or no exercise
        'light': 1.375,          # Light exercise 1-3 days/week
        'moderate': 1.55,        # Moderate exercise 3-5 days/week
        'very_active': 1.725,    # Hard exercise 6-7 days/week
        'extra_active': 1.9      # Very hard exercise / physical job
    }
    multiplier = activity_multipliers.get(activity_level.lower(), 1.375)
    tdee = round(bmr * multiplier)

    # Goal-based caloric adjustments
    if goal.lower() == 'weight_loss':
        target_calories = max(1200, tdee - 450)
        protein_per_kg = 1.8  # Higher protein to preserve lean muscle
        fat_ratio = 0.25
    elif goal.lower() == 'muscle_gain':
        target_calories = tdee + 350
        protein_per_kg = 2.0  # Surplus + high protein for muscle synthesis
        fat_ratio = 0.25
    else:  # maintenance
        target_calories = tdee
        protein_per_kg = 1.5
        fat_ratio = 0.28

    # Macro breakdown
    protein_g = round(weight_kg * protein_per_kg)
    protein_cal = protein_g * 4
    
    fat_cal = target_calories * fat_ratio
    fat_g = round(fat_cal / 9)
    
    carbs_cal = max(0, target_calories - protein_cal - fat_cal)
    carbs_g = round(carbs_cal / 4)

    return {
        "bmr": bmr,
        "tdee": tdee,
        "target_calories": round(target_calories),
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fat_g": fat_g
    }


# -------------------------------------------------------------------------
# 2. DIET RECOMMENDATION (Content-Based ML + Constraint Filters)
# -------------------------------------------------------------------------

def filter_foods(foods_df, dietary_pref, user_allergies):
    """
    Apply strict constraints:
    1. Dietary preference (Veg, Non-Veg, Vegan, Eggetarian)
    2. Zero-tolerance Food Allergies filter (never relaxed)
    """
    pref = dietary_pref.strip().lower()
    
    # 1. Diet Type Filter
    if pref == 'vegan':
        filtered = foods_df[foods_df['diet_type'].str.lower() == 'vegan'].copy()
    elif pref == 'veg':
        filtered = foods_df[foods_df['diet_type'].str.lower().isin(['veg', 'vegan'])].copy()
    elif pref == 'eggetarian':
        filtered = foods_df[foods_df['diet_type'].str.lower().isin(['veg', 'vegan', 'eggetarian'])].copy()
    else:  # non-veg
        filtered = foods_df.copy()

    # 2. Strict Allergen Filtering
    if user_allergies:
        clean_user_allergies = [a.strip().lower() for a in user_allergies if a.strip()]
        if clean_user_allergies:
            def contains_allergen(food_allergens_str):
                if pd.isna(food_allergens_str) or food_allergens_str.strip().lower() in ['none', '']:
                    return False
                # allergens in csv are separated by ';'
                item_allergens = [x.strip().lower() for x in str(food_allergens_str).split(';') if x.strip()]
                return any(ua in item_allergens for ua in clean_user_allergies)

            # Keep only items that do NOT contain any user allergens
            filtered = filtered[~filtered['allergens'].apply(contains_allergen)]

    return filtered


def rank_foods_with_ml(candidate_df, target_meal_vector, n_recommendations=6):
    """
    Content-Based Recommendation using NearestNeighbors with Cosine Distance.
    Features: [calories, protein_g, carbs_g, fat_g, fiber_g]
    """
    if len(candidate_df) == 0:
        return []

    feature_cols = ['calories', 'protein_g', 'carbs_g', 'fat_g', 'fiber_g']
    X = candidate_df[feature_cols].values

    # Feature Scaling using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale target vector using the same fitted scaler
    target_scaled = scaler.transform([target_meal_vector])

    # K-Nearest Neighbors model with cosine distance
    k = min(n_recommendations, len(candidate_df))
    nn_model = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    nn_model.fit(X_scaled)

    distances, indices = nn_model.kneighbors(target_scaled)

    ranked_items = []
    for dist, idx in zip(distances[0], indices[0]):
        item = candidate_df.iloc[idx].to_dict()
        # Cosine similarity score = 1 - cosine_distance (bounded 0 to 1)
        similarity = round(max(0.0, 1.0 - dist) * 100, 1)
        item['similarity_score'] = similarity
        ranked_items.append(item)

    return ranked_items


def generate_7day_meal_plan(foods_df, target_calories, target_protein, target_carbs, target_fat, dietary_pref, user_allergies):
    """
    Generates a structured 7-day meal plan.
    - Breakfast: 25% daily target
    - Lunch: 35% daily target
    - Snack: 10% daily target
    - Dinner: 30% daily target
    Uses NearestNeighbors to rank best candidates per meal slot, then applies rule-based portion scaling and daily variations.
    """
    filtered_foods = filter_foods(foods_df, dietary_pref, user_allergies)

    # Check availability for each meal type
    meal_types = ['breakfast', 'lunch', 'snack', 'dinner']
    meal_splits = {
        'breakfast': 0.25,
        'lunch': 0.35,
        'snack': 0.10,
        'dinner': 0.30
    }

    candidates_by_meal = {}
    for mtype in meal_types:
        subset = filtered_foods[filtered_foods['meal_type'].str.lower() == mtype]
        if len(subset) == 0:
            allergens_str = ", ".join(user_allergies) if user_allergies else "None"
            raise ValueError(
                f"No suitable foods found for meal '{mtype.capitalize()}' matching dietary preference '{dietary_pref}' "
                f"and allergy constraints [{allergens_str}]. Please adjust your allergy filters or dietary preference."
            )
        candidates_by_meal[mtype] = subset

    # ML Ranking per meal slot
    ranked_by_meal = {}
    for mtype, ratio in meal_splits.items():
        slot_target = [
            target_calories * ratio,
            target_protein * ratio,
            target_carbs * ratio,
            target_fat * ratio,
            8.0 * ratio  # target fiber benchmark
        ]
        ranked_by_meal[mtype] = rank_foods_with_ml(candidates_by_meal[mtype], slot_target, n_recommendations=7)

    # Build 7-day schedule with variety and rule-based portion scaling
    day_names = ["Day 1 (Monday)", "Day 2 (Tuesday)", "Day 3 (Wednesday)", "Day 4 (Thursday)", "Day 5 (Friday)", "Day 6 (Saturday)", "Day 7 (Sunday)"]
    weekly_plan = []

    for day_idx, day_name in enumerate(day_names):
        day_meals = []
        day_cal_total = 0
        day_protein_total = 0
        day_carbs_total = 0
        day_fat_total = 0

        for mtype in meal_types:
            pool = ranked_by_meal[mtype]
            # Rotate through top recommendations to avoid repetitive daily meals
            selected_item = pool[day_idx % len(pool)].copy()

            # Rule-based portion scaling (multiplier to match target meal calories)
            target_meal_cal = target_calories * meal_splits[mtype]
            raw_scale = target_meal_cal / max(100.0, selected_item['calories'])
            portion_scale = round(min(1.4, max(0.7, raw_scale)), 2)

            scaled_cal = round(selected_item['calories'] * portion_scale)
            scaled_protein = round(selected_item['protein_g'] * portion_scale, 1)
            scaled_carbs = round(selected_item['carbs_g'] * portion_scale, 1)
            scaled_fat = round(selected_item['fat_g'] * portion_scale, 1)
            scaled_weight = round(selected_item['default_serving_weight_g'] * portion_scale)

            portion_desc = f"{portion_scale}x serving (~{scaled_weight}g / {selected_item['default_serving_unit']})"

            meal_entry = {
                "meal_type": mtype.capitalize(),
                "name": selected_item['name'],
                "portion": portion_desc,
                "calories": scaled_cal,
                "protein_g": scaled_protein,
                "carbs_g": scaled_carbs,
                "fat_g": scaled_fat,
                "allergens": selected_item.get('allergens', 'none'),
                "similarity_score": selected_item.get('similarity_score', 95.0)
            }
            day_meals.append(meal_entry)
            day_cal_total += scaled_cal
            day_protein_total += scaled_protein
            day_carbs_total += scaled_carbs
            day_fat_total += scaled_fat

        weekly_plan.append({
            "day": day_name,
            "meals": day_meals,
            "total_calories": day_cal_total,
            "total_protein": round(day_protein_total, 1),
            "total_carbs": round(day_carbs_total, 1),
            "total_fat": round(day_fat_total, 1)
        })

    return weekly_plan


# -------------------------------------------------------------------------
# 3. WORKOUT RECOMMENDATION (Content-Based ML + Rule-Based Split Generator)
# -------------------------------------------------------------------------

def filter_workouts(workouts_df, equipment, experience):
    """
    Filter workout exercises by equipment availability and experience level.
    """
    eq = equipment.strip().lower()
    exp = experience.strip().lower()

    # 1. Equipment hierarchy
    if eq == 'bodyweight':
        filtered = workouts_df[workouts_df['equipment_required'].str.lower() == 'bodyweight'].copy()
    elif eq == 'dumbbells':
        filtered = workouts_df[workouts_df['equipment_required'].str.lower().isin(['bodyweight', 'dumbbells'])].copy()
    else:  # gym
        filtered = workouts_df.copy()

    # 2. Experience level compatibility
    if exp == 'beginner':
        filtered = filtered[filtered['difficulty_level'].str.lower().isin(['beginner', 'intermediate'])].copy()
    elif exp == 'intermediate':
        filtered = filtered[filtered['difficulty_level'].str.lower().isin(['beginner', 'intermediate', 'advanced'])].copy()
    else:  # advanced
        filtered = filtered.copy()

    if len(filtered) < 4:
        # Fallback to entire equipment pool if filtered is too sparse
        if eq == 'bodyweight':
            filtered = workouts_df[workouts_df['equipment_required'].str.lower() == 'bodyweight'].copy()
        elif eq == 'dumbbells':
            filtered = workouts_df[workouts_df['equipment_required'].str.lower().isin(['bodyweight', 'dumbbells'])].copy()
        else:
            filtered = workouts_df.copy()

    return filtered


def rank_workouts_with_ml(candidate_df, target_intensity_vector, n_recommendations=10):
    """
    Content-Based ML ranking for workouts using NearestNeighbors on intensity & calorie burn.
    Features: [intensity_score, calorie_burn_per_min]
    """
    if len(candidate_df) == 0:
        return []

    feature_cols = ['intensity_score', 'calorie_burn_per_min']
    X = candidate_df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    target_scaled = scaler.transform([target_intensity_vector])

    k = min(n_recommendations, len(candidate_df))
    nn_model = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    nn_model.fit(X_scaled)

    distances, indices = nn_model.kneighbors(target_scaled)

    ranked_items = []
    for dist, idx in zip(distances[0], indices[0]):
        item = candidate_df.iloc[idx].to_dict()
        similarity = round(max(0.0, 1.0 - dist) * 100, 1)
        item['similarity_score'] = similarity
        ranked_items.append(item)

    return ranked_items


def generate_weekly_workout_plan(workouts_df, goal, experience, equipment, workout_days):
    """
    Generates a personalized weekly workout schedule based on ML ranking and split templates.
    """
    filtered_workouts = filter_workouts(workouts_df, equipment, experience)

    # Define target ML vector based on goal and experience
    if goal.lower() == 'weight_loss':
        target_vector = [7.5, 8.5]  # Higher metabolic demand
    elif goal.lower() == 'muscle_gain':
        target_vector = [8.0, 7.0]  # Progressive resistance & hypertrophy
    else:  # maintenance
        target_vector = [6.5, 6.0]  # Balanced general fitness

    # Split structure by number of days
    days = int(workout_days)
    if days <= 3:
        split_schedule = [
            {"day": "Day 1 (Monday)", "type": "Full Body Focus A", "is_rest": False, "target_categories": ["Push", "Pull", "Legs", "Core"]},
            {"day": "Day 2 (Tuesday)", "type": "Active Recovery / Rest", "is_rest": True, "target_categories": []},
            {"day": "Day 3 (Wednesday)", "type": "Full Body Focus B", "is_rest": False, "target_categories": ["Push", "Pull", "Legs", "Cardio"]},
            {"day": "Day 4 (Thursday)", "type": "Active Recovery / Rest", "is_rest": True, "target_categories": []},
            {"day": "Day 5 (Friday)", "type": "Full Body & Core C", "is_rest": False, "target_categories": ["Push", "Pull", "Legs", "Core"]},
            {"day": "Day 6 (Saturday)", "type": "Active Recovery / Rest", "is_rest": True, "target_categories": []},
            {"day": "Day 7 (Sunday)", "type": "Full Rest & Mobility", "is_rest": True, "target_categories": []}
        ]
    elif days == 4:
        split_schedule = [
            {"day": "Day 1 (Monday)", "type": "Upper Body Strength (Push & Pull)", "is_rest": False, "target_categories": ["Push", "Pull"]},
            {"day": "Day 2 (Tuesday)", "type": "Lower Body & Core", "is_rest": False, "target_categories": ["Legs", "Core"]},
            {"day": "Day 3 (Wednesday)", "type": "Active Recovery / Rest", "is_rest": True, "target_categories": []},
            {"day": "Day 4 (Thursday)", "type": "Upper Body Hypertrophy", "is_rest": False, "target_categories": ["Push", "Pull"]},
            {"day": "Day 5 (Friday)", "type": "Lower Body & Cardio Burn", "is_rest": False, "target_categories": ["Legs", "Cardio", "Core"]},
            {"day": "Day 6 (Saturday)", "type": "Active Recovery / Rest", "is_rest": True, "target_categories": []},
            {"day": "Day 7 (Sunday)", "type": "Full Rest & Mobility", "is_rest": True, "target_categories": []}
        ]
    elif days == 5:
        split_schedule = [
            {"day": "Day 1 (Monday)", "type": "Chest & Triceps (Push Focus)", "is_rest": False, "target_categories": ["Push"]},
            {"day": "Day 2 (Tuesday)", "type": "Back & Biceps (Pull Focus)", "is_rest": False, "target_categories": ["Pull"]},
            {"day": "Day 3 (Wednesday)", "type": "Legs & Calves (Lower Focus)", "is_rest": False, "target_categories": ["Legs"]},
            {"day": "Day 4 (Thursday)", "type": "Shoulders, Arms & Core", "is_rest": False, "target_categories": ["Push", "Core"]},
            {"day": "Day 5 (Friday)", "type": "Full Body Conditioning & Cardio", "is_rest": False, "target_categories": ["Full Body", "Cardio", "Core"]},
            {"day": "Day 6 (Saturday)", "type": "Active Recovery / Walk", "is_rest": True, "target_categories": []},
            {"day": "Day 7 (Sunday)", "type": "Full Rest & Regeneration", "is_rest": True, "target_categories": []}
        ]
    else:  # 6 days
        split_schedule = [
            {"day": "Day 1 (Monday)", "type": "Push A (Chest, Shoulders, Triceps)", "is_rest": False, "target_categories": ["Push"]},
            {"day": "Day 2 (Tuesday)", "type": "Pull A (Back, Biceps, Rear Delts)", "is_rest": False, "target_categories": ["Pull"]},
            {"day": "Day 3 (Wednesday)", "type": "Legs A (Quads, Hamstrings, Calves)", "is_rest": False, "target_categories": ["Legs", "Core"]},
            {"day": "Day 4 (Thursday)", "type": "Push B (Shoulder & Chest Focus)", "is_rest": False, "target_categories": ["Push"]},
            {"day": "Day 5 (Friday)", "type": "Pull B (Lat & Arm Hypertrophy)", "is_rest": False, "target_categories": ["Pull"]},
            {"day": "Day 6 (Saturday)", "type": "Legs B & Core Conditioning", "is_rest": False, "target_categories": ["Legs", "Cardio", "Core"]},
            {"day": "Day 7 (Sunday)", "type": "Full Rest & Recovery", "is_rest": True, "target_categories": []}
        ]

    # Populate exercises for active workout days
    weekly_workout_plan = []
    exercise_rotation_index = 0

    for day_info in split_schedule:
        if day_info["is_rest"]:
            weekly_workout_plan.append({
                "day": day_info["day"],
                "focus": day_info["type"],
                "is_rest": True,
                "notes": "Perform 20-30 minutes of light walking, gentle stretching, and foam rolling. Focus on sleep and hydration.",
                "exercises": []
            })
            continue

        day_exercises = []
        categories = day_info["target_categories"]

        # Gather exercises for requested categories
        pool_for_day = filtered_workouts[filtered_workouts['category'].isin(categories)]
        if len(pool_for_day) == 0:
            pool_for_day = filtered_workouts  # Fallback

        # Rank with NearestNeighbors
        ranked = rank_workouts_with_ml(pool_for_day, target_vector, n_recommendations=8)

        # Select 4-5 distinct exercises
        selected_count = min(5, len(ranked))
        for i in range(selected_count):
            item = ranked[(i + exercise_rotation_index) % len(ranked)]
            
            # Goal-based sets & reps adjustment rule
            if goal.lower() == 'muscle_gain':
                sets = item.get('default_sets', 3)
                reps = "8-12 reps (Hypertrophy focus)"
            elif goal.lower() == 'weight_loss':
                sets = item.get('default_sets', 3)
                reps = "12-15 reps (Endurance & High Burn)"
            else:
                sets = item.get('default_sets', 3)
                reps = item.get('default_reps', '10-12 reps')

            day_exercises.append({
                "name": item['name'],
                "target_muscle": item['target_muscle'],
                "equipment": item['equipment_required'].capitalize(),
                "sets": sets,
                "reps": reps,
                "rest": f"{item['rest_seconds']}s",
                "instructions": item['instructions'],
                "form_cues": item['form_cues'],
                "home_alternative": item.get('home_alternative', 'Bodyweight modification')
            })

        exercise_rotation_index += 1
        weekly_workout_plan.append({
            "day": day_info["day"],
            "focus": day_info["type"],
            "is_rest": False,
            "notes": f"Warm up for 5 minutes before starting. Target 45-60 min session. Rest {item['rest_seconds']}s between sets.",
            "exercises": day_exercises
        })

    return weekly_workout_plan
