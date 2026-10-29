"""
tests/test_system.py
====================
Unit and Integration tests for:
1. Physical biometrics & nutritional calculation rules (BMI, BMR, TDEE)
2. Dietary & strict allergy filtering constraints
3. Scikit-learn NearestNeighbors Content-Based Recommendation outputs
4. SQLite persistence operations (Save, List, Fetch, Delete)
5. Flask Web Application routes and endpoints (Validation, Dashboard, Export)
"""

import unittest
import json
import os
import sys

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ml_engine import (
    load_datasets,
    calculate_bmi,
    calculate_bmr,
    calculate_tdee_and_targets,
    filter_foods,
    rank_foods_with_ml,
    generate_7day_meal_plan,
    filter_workouts,
    rank_workouts_with_ml,
    generate_weekly_workout_plan
)
import database
from app import app


class NutriFitSystemTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.foods_df, cls.workouts_df = load_datasets()
        database.init_db()

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    # ---------------------------------------------------------------------
    # 1. BIOMETRIC AND NUTRITION FORMULA TESTS
    # ---------------------------------------------------------------------
    def test_bmi_calculation(self):
        # 70 kg, 175 cm -> BMI = 70 / (1.75^2) = 22.86 -> 22.9
        result = calculate_bmi(70, 175)
        self.assertEqual(result['bmi'], 22.9)
        self.assertEqual(result['category'], 'Normal weight')

        # 95 kg, 170 cm -> Overweight / Obese
        obese_result = calculate_bmi(95, 170)
        self.assertGreaterEqual(obese_result['bmi'], 30.0)
        self.assertEqual(obese_result['category'], 'Obese')

    def test_bmr_and_tdee_targets(self):
        # Male, 25 yrs, 175 cm, 70 kg
        tdee_data = calculate_tdee_and_targets(
            weight_kg=70, height_cm=175, age=25, gender='male',
            activity_level='moderate', goal='weight_loss'
        )
        self.assertIn('bmr', tdee_data)
        self.assertIn('tdee', tdee_data)
        self.assertIn('target_calories', tdee_data)
        self.assertIn('protein_g', tdee_data)
        # In weight loss, target calories should be less than TDEE
        self.assertLess(tdee_data['target_calories'], tdee_data['tdee'])
        self.assertGreater(tdee_data['protein_g'], 0)

    # ---------------------------------------------------------------------
    # 2. ALLERGY FILTERING AND CONSTRAINT TESTS
    # ---------------------------------------------------------------------
    def test_strict_allergy_filtering_dairy(self):
        # User allergic to dairy
        filtered = filter_foods(self.foods_df, dietary_pref='veg', user_allergies=['dairy'])
        # Check that none of the returned foods contain dairy
        for _, row in filtered.iterrows():
            allergens = str(row['allergens']).lower()
            self.assertNotIn('dairy', allergens)

    def test_strict_allergy_filtering_peanuts_and_gluten(self):
        # User allergic to peanuts and gluten
        filtered = filter_foods(self.foods_df, dietary_pref='vegan', user_allergies=['peanuts', 'gluten'])
        for _, row in filtered.iterrows():
            allergens = str(row['allergens']).lower()
            self.assertNotIn('peanuts', allergens)
            self.assertNotIn('gluten', allergens)

    def test_strict_allergy_impossible_combination_raises_error(self):
        # When a meal slot has 0 items due to overly restrictive filters, ValueError is raised
        empty_foods_df = self.foods_df[self.foods_df['diet_type'] == 'nonexistent']
        with self.assertRaises(ValueError):
            generate_7day_meal_plan(
                foods_df=empty_foods_df,
                target_calories=2000,
                target_protein=120,
                target_carbs=200,
                target_fat=50,
                dietary_pref='veg',
                user_allergies=[]
            )

    # ---------------------------------------------------------------------
    # 3. CONTENT-BASED ML RECOMMENDATION TESTS
    # ---------------------------------------------------------------------
    def test_meal_plan_generation_structure(self):
        meal_plan = generate_7day_meal_plan(
            foods_df=self.foods_df,
            target_calories=1800,
            target_protein=130,
            target_carbs=180,
            target_fat=50,
            dietary_pref='veg',
            user_allergies=[]
        )
        self.assertEqual(len(meal_plan), 7)
        for day in meal_plan:
            self.assertIn('day', day)
            self.assertIn('meals', day)
            self.assertEqual(len(day['meals']), 4)  # Breakfast, Lunch, Snack, Dinner
            self.assertGreater(day['total_calories'], 1000)

    def test_workout_plan_generation_structure(self):
        workout_plan = generate_weekly_workout_plan(
            workouts_df=self.workouts_df,
            goal='muscle_gain',
            experience='beginner',
            equipment='dumbbells',
            workout_days=4
        )
        self.assertEqual(len(workout_plan), 7)
        # Should contain active days and rest days
        active_days = [d for d in workout_plan if not d['is_rest']]
        rest_days = [d for d in workout_plan if d['is_rest']]
        self.assertEqual(len(active_days), 4)
        self.assertEqual(len(rest_days), 3)

        # Check exercise content in active days
        for day in active_days:
            self.assertGreater(len(day['exercises']), 0)
            for ex in day['exercises']:
                self.assertIn('name', ex)
                self.assertIn('sets', ex)
                self.assertIn('reps', ex)
                self.assertIn('form_cues', ex)
                self.assertIn('home_alternative', ex)

    # ---------------------------------------------------------------------
    # 4. DATABASE PERSISTENCE TESTS
    # ---------------------------------------------------------------------
    def test_save_and_retrieve_plan(self):
        sample_plan = {
            "age": 22,
            "gender": "male",
            "height": 180.0,
            "weight": 75.0,
            "activity_level": "moderate",
            "fitness_goal": "muscle_gain",
            "dietary_pref": "veg",
            "allergies": ["peanuts"],
            "workout_experience": "intermediate",
            "available_equipment": "dumbbells",
            "workout_days": 4,
            "bmi": {"bmi": 23.1, "category": "Normal weight"},
            "nutrition": {"target_calories": 2400, "protein_g": 150, "carbs_g": 260, "fat_g": 65},
            "meal_plan": [{"day": "Day 1", "meals": [], "total_calories": 2400, "total_protein": 150, "total_carbs": 260, "total_fat": 65}],
            "workout_plan": [{"day": "Day 1", "focus": "Upper Body", "is_rest": False, "notes": "", "exercises": []}]
        }
        plan_id = database.save_plan(sample_plan)
        self.assertIsNotNone(plan_id)

        retrieved = database.get_plan_by_id(plan_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved['age'], 22)
        self.assertEqual(retrieved['target_calories'], 2400)
        self.assertEqual(len(retrieved['meal_plan']), 1)

        # Cleanup
        database.delete_plan_by_id(plan_id)
        self.assertIsNone(database.get_plan_by_id(plan_id))

    # ---------------------------------------------------------------------
    # 5. FLASK ROUTE INTEGRATION TESTS
    # ---------------------------------------------------------------------
    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'User Profile Questionnaire', response.data)

    def test_underage_validation(self):
        response = self.app.post('/generate', data={
            'age': '16',  # < 18
            'gender': 'male',
            'height': '170',
            'weight': '60',
            'activity_level': 'light',
            'fitness_goal': 'weight_loss',
            'dietary_pref': 'veg',
            'workout_experience': 'beginner',
            'available_equipment': 'bodyweight',
            'workout_days': '3'
        }, follow_redirects=True)
        self.assertIn(b'adults (age 18+)', response.data)

    def test_valid_generation_route(self):
        response = self.app.post('/generate', data={
            'age': '24',
            'gender': 'male',
            'height': '175',
            'weight': '70',
            'activity_level': 'moderate',
            'fitness_goal': 'weight_loss',
            'dietary_pref': 'veg',
            'allergies': ['peanuts'],
            'workout_experience': 'beginner',
            'available_equipment': 'dumbbells',
            'workout_days': '4'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Recommendation Dashboard', response.data)
        self.assertIn(b'7-Day Nutritional Schedule', response.data)

    def test_download_plan_txt_export(self):
        # Generate and save a plan first
        sample_plan = {
            "age": 28,
            "gender": "female",
            "height": 165.0,
            "weight": 58.0,
            "activity_level": "moderate",
            "fitness_goal": "maintenance",
            "dietary_pref": "veg",
            "allergies": ["none"],
            "workout_experience": "beginner",
            "available_equipment": "bodyweight",
            "workout_days": 3,
            "bmi": {"bmi": 21.3, "category": "Normal weight"},
            "nutrition": {"target_calories": 1800, "protein_g": 90, "carbs_g": 220, "fat_g": 50},
            "meal_plan": [{"day": "Day 1 (Monday)", "meals": [{"meal_type": "Breakfast", "name": "Oats", "portion": "1 bowl", "calories": 300, "protein_g": 12, "carbs_g": 45, "fat_g": 6, "allergens": "none", "similarity_score": 98.0}], "total_calories": 1800, "total_protein": 90, "total_carbs": 220, "total_fat": 50}],
            "workout_plan": [{"day": "Day 1 (Monday)", "focus": "Full Body", "is_rest": False, "notes": "Warm up", "exercises": [{"name": "Push-ups", "target_muscle": "Chest", "equipment": "Bodyweight", "sets": 3, "reps": "10 reps", "rest": "60s", "form_cues": "Keep flat", "home_alternative": "Wall pushups"}]}]
        }
        plan_id = database.save_plan(sample_plan, is_saved=1)
        
        # Test download endpoint
        response = self.app.get(f'/download/{plan_id}')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'text/plain; charset=utf-8')
        self.assertIn(b'PERSONALIZED DIET AND WORKOUT RECOMMENDATION SYSTEM', response.data)
        self.assertIn(b'7-DAY PERSONALIZED MEAL PLAN', response.data)
        self.assertIn(b'WEEKLY WORKOUT SCHEDULE', response.data)

        # Test saved plans list page
        saved_list_resp = self.app.get('/saved')
        self.assertEqual(saved_list_resp.status_code, 200)
        self.assertIn(b'Saved Recommendation Plans', saved_list_resp.data)

        # Test delete endpoint
        del_resp = self.app.post(f'/delete/{plan_id}', follow_redirects=True)
        self.assertEqual(del_resp.status_code, 200)
        self.assertIsNone(database.get_plan_by_id(plan_id))


if __name__ == '__main__':
    unittest.main()

