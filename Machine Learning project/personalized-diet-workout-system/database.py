"""
database.py
===========
SQLite persistence layer for saving and retrieving generated diet & workout plans.
"""

import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')


def get_db_connection():
    """Create a database connection with dict-like row access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS saved_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_saved INTEGER DEFAULT 1,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            height REAL NOT NULL,
            weight REAL NOT NULL,
            activity_level TEXT NOT NULL,
            fitness_goal TEXT NOT NULL,
            dietary_pref TEXT NOT NULL,
            allergies TEXT,
            workout_experience TEXT NOT NULL,
            available_equipment TEXT NOT NULL,
            workout_days INTEGER NOT NULL,
            bmi REAL NOT NULL,
            bmi_category TEXT NOT NULL,
            target_calories INTEGER NOT NULL,
            protein_g INTEGER NOT NULL,
            carbs_g INTEGER NOT NULL,
            fat_g INTEGER NOT NULL,
            meal_plan_json TEXT NOT NULL,
            workout_plan_json TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def save_plan(plan_data, is_saved=1):
    """
    Save a full generated plan into SQLite.
    Returns the newly created plan ID.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    allergies_str = ", ".join(plan_data.get('allergies', [])) if isinstance(plan_data.get('allergies'), list) else plan_data.get('allergies', 'None')
    meal_json = json.dumps(plan_data['meal_plan']) if not isinstance(plan_data['meal_plan'], str) else plan_data['meal_plan']
    workout_json = json.dumps(plan_data['workout_plan']) if not isinstance(plan_data['workout_plan'], str) else plan_data['workout_plan']

    cursor.execute('''
        INSERT INTO saved_plans (
            is_saved, age, gender, height, weight, activity_level, fitness_goal,
            dietary_pref, allergies, workout_experience, available_equipment,
            workout_days, bmi, bmi_category, target_calories,
            protein_g, carbs_g, fat_g, meal_plan_json, workout_plan_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        is_saved,
        plan_data['age'],
        plan_data['gender'],
        plan_data['height'],
        plan_data['weight'],
        plan_data['activity_level'],
        plan_data['fitness_goal'],
        plan_data['dietary_pref'],
        allergies_str,
        plan_data['workout_experience'],
        plan_data['available_equipment'],
        plan_data['workout_days'],
        plan_data['bmi']['bmi'],
        plan_data['bmi']['category'],
        plan_data['nutrition']['target_calories'],
        plan_data['nutrition']['protein_g'],
        plan_data['nutrition']['carbs_g'],
        plan_data['nutrition']['fat_g'],
        meal_json,
        workout_json
    ))
    plan_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return plan_id


def mark_plan_as_saved(plan_id):
    """Mark an existing plan as saved/bookmarked."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE saved_plans SET is_saved = 1 WHERE id = ?', (plan_id,))
    conn.commit()
    conn.close()


def get_all_saved_plans():
    """Fetch summaries of all bookmarked saved plans."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, created_at, age, gender, height, weight, fitness_goal,
               dietary_pref, allergies, workout_experience, available_equipment,
               workout_days, bmi, bmi_category, target_calories, protein_g
        FROM saved_plans
        WHERE is_saved = 1
        ORDER BY id DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_plan_by_id(plan_id):
    """Retrieve full details of a specific saved plan."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM saved_plans WHERE id = ?', (plan_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None

    plan = dict(row)
    plan['meal_plan'] = json.loads(plan['meal_plan_json'])
    plan['workout_plan'] = json.loads(plan['workout_plan_json'])
    return plan


def delete_plan_by_id(plan_id):
    """Delete a plan by ID."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM saved_plans WHERE id = ?', (plan_id,))
    conn.commit()
    conn.close()
