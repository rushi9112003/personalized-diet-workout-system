"""
app.py
======
Flask Web Application for the Personalized Diet and Workout Recommendation System.
"""

from flask import Flask, render_template, request, redirect, url_for, flash, session, Response
import os
import json
from ml_engine import (
    load_datasets,
    calculate_bmi,
    calculate_tdee_and_targets,
    generate_7day_meal_plan,
    generate_weekly_workout_plan
)
import database

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize database on startup
database.init_db()

# Pre-load datasets
foods_df, workouts_df = load_datasets()


@app.route('/')
def index():
    """Render the user input form."""
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """Process user inputs, perform ML ranking, and display dashboard."""
    try:
        # 1. Parse and validate form inputs
        age = int(request.form.get('age', 0))
        gender = request.form.get('gender', 'male')
        height = float(request.form.get('height', 0))
        weight = float(request.form.get('weight', 0))
        activity_level = request.form.get('activity_level', 'moderate')
        fitness_goal = request.form.get('fitness_goal', 'weight_loss')
        dietary_pref = request.form.get('dietary_pref', 'veg')
        allergies = request.form.getlist('allergies')
        workout_experience = request.form.get('workout_experience', 'beginner')
        available_equipment = request.form.get('available_equipment', 'bodyweight')
        workout_days = int(request.form.get('workout_days', 4))

        # Input Validations
        if age < 18:
            flash("This system is designed for adults (age 18+). Please enter an age of 18 or above.", "danger")
            return redirect(url_for('index'))
        if age > 100:
            flash("Please enter a realistic age (18 - 100).", "danger")
            return redirect(url_for('index'))
        if height < 100 or height > 250:
            flash("Please enter a valid height in centimeters (100 cm - 250 cm).", "danger")
            return redirect(url_for('index'))
        if weight < 30 or weight > 250:
            flash("Please enter a valid weight in kilograms (30 kg - 250 kg).", "danger")
            return redirect(url_for('index'))

        # 2. Biometrics Calculations
        bmi_data = calculate_bmi(weight, height)
        nutrition_data = calculate_tdee_and_targets(weight, height, age, gender, activity_level, fitness_goal)

        # 3. Content-Based ML Recommendations
        meal_plan = generate_7day_meal_plan(
            foods_df=foods_df,
            target_calories=nutrition_data['target_calories'],
            target_protein=nutrition_data['protein_g'],
            target_carbs=nutrition_data['carbs_g'],
            target_fat=nutrition_data['fat_g'],
            dietary_pref=dietary_pref,
            user_allergies=allergies
        )

        workout_plan = generate_weekly_workout_plan(
            workouts_df=workouts_df,
            goal=fitness_goal,
            experience=workout_experience,
            equipment=available_equipment,
            workout_days=workout_days
        )

        # 4. Package plan and save initial generated record to database
        plan_data = {
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "activity_level": activity_level,
            "fitness_goal": fitness_goal,
            "dietary_pref": dietary_pref,
            "allergies": allergies,
            "workout_experience": workout_experience,
            "available_equipment": available_equipment,
            "workout_days": workout_days,
            "bmi": bmi_data,
            "nutrition": nutrition_data,
            "meal_plan": meal_plan,
            "workout_plan": workout_plan
        }

        plan_id = database.save_plan(plan_data, is_saved=0)
        plan_data["id"] = plan_id

        return render_template('dashboard.html', plan=plan_data, is_saved_view=False)

    except ValueError as ve:
        flash(str(ve), "warning")
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"An unexpected error occurred during plan generation: {str(e)}", "danger")
        return redirect(url_for('index'))


@app.route('/save/<int:plan_id>', methods=['POST'])
def save(plan_id):
    """Mark the active generated plan as bookmarked in SQLite database."""
    try:
        database.mark_plan_as_saved(plan_id)
        flash(f"Plan #{plan_id} successfully saved to your Saved Plans bookmark list!", "success")
        return redirect(url_for('view_plan', plan_id=plan_id))
    except Exception as e:
        flash(f"Failed to save plan: {str(e)}", "danger")
        return redirect(url_for('index'))


@app.route('/saved')
def saved_plans():
    """List all saved plans."""
    plans = database.get_all_saved_plans()
    return render_template('saved_plans.html', plans=plans)


@app.route('/plan/<int:plan_id>')
def view_plan(plan_id):
    """View details of a specific saved plan."""
    plan_row = database.get_plan_by_id(plan_id)
    if not plan_row:
        flash(f"Plan #{plan_id} not found.", "danger")
        return redirect(url_for('saved_plans'))

    # Construct plan_data structure for template
    plan_data = {
        "id": plan_row['id'],
        "created_at": plan_row['created_at'],
        "is_saved": plan_row['is_saved'],
        "age": plan_row['age'],
        "gender": plan_row['gender'],
        "height": plan_row['height'],
        "weight": plan_row['weight'],
        "activity_level": plan_row['activity_level'],
        "fitness_goal": plan_row['fitness_goal'],
        "dietary_pref": plan_row['dietary_pref'],
        "allergies": [a.strip() for a in plan_row['allergies'].split(',') if a.strip() and a.strip() != 'None'],
        "workout_experience": plan_row['workout_experience'],
        "available_equipment": plan_row['available_equipment'],
        "workout_days": plan_row['workout_days'],
        "bmi": {
            "bmi": plan_row['bmi'],
            "category": plan_row['bmi_category'],
            "explanation": "Calculated BMI for your biometrics.",
            "badge_class": "success" if plan_row['bmi_category'] == 'Normal weight' else "warning"
        },
        "nutrition": {
            "target_calories": plan_row['target_calories'],
            "protein_g": plan_row['protein_g'],
            "carbs_g": plan_row['carbs_g'],
            "fat_g": plan_row['fat_g']
        },
        "meal_plan": plan_row['meal_plan'],
        "workout_plan": plan_row['workout_plan']
    }

    return render_template('dashboard.html', plan=plan_data, is_saved_view=(plan_row['is_saved'] == 1))


@app.route('/download/<int:plan_id>')
def download_plan(plan_id):
    """Download plan as a cleanly formatted text file."""
    plan = database.get_plan_by_id(plan_id)
    if not plan:
        flash("Plan not found for download.", "danger")
        return redirect(url_for('saved_plans'))

    # Format text content
    lines = []
    lines.append("=" * 70)
    lines.append("   PERSONALIZED DIET AND WORKOUT RECOMMENDATION SYSTEM")
    lines.append(f"   Saved Plan #{plan['id']} | Generated: {plan['created_at']}")
    lines.append("=" * 70)
    lines.append("\n[1] USER PROFILE & BIOMETRICS")
    lines.append("-" * 50)
    lines.append(f"• Age: {plan['age']} years | Gender: {plan['gender'].capitalize()}")
    lines.append(f"• Height: {plan['height']} cm | Weight: {plan['weight']} kg")
    lines.append(f"• BMI: {plan['bmi']} ({plan['bmi_category']})")
    lines.append(f"• Activity Level: {plan['activity_level'].replace('_', ' ').capitalize()}")
    lines.append(f"• Fitness Goal: {plan['fitness_goal'].replace('_', ' ').capitalize()}")
    lines.append(f"• Dietary Preference: {plan['dietary_pref'].capitalize()}")
    lines.append(f"• Food Allergies Filtered: {plan['allergies']}")
    lines.append(f"• Experience: {plan['workout_experience'].capitalize()} | Equipment: {plan['available_equipment'].capitalize()}")
    lines.append(f"• Workout Schedule: {plan['workout_days']} days per week")

    lines.append("\n[2] DAILY TARGET NUTRITION")
    lines.append("-" * 50)
    lines.append(f"• Daily Target Energy: {plan['target_calories']} kcal")
    lines.append(f"• Protein Target:      {plan['protein_g']} g ({round(plan['protein_g']*4*100/plan['target_calories'])}% of calories)")
    lines.append(f"• Carbohydrates:       {plan['carbs_g']} g ({round(plan['carbs_g']*4*100/plan['target_calories'])}% of calories)")
    lines.append(f"• Healthy Fats:        {plan['fat_g']} g ({round(plan['fat_g']*9*100/plan['target_calories'])}% of calories)")

    lines.append("\n[3] 7-DAY PERSONALIZED MEAL PLAN (ML Ranked)")
    lines.append("=" * 70)
    for day in plan['meal_plan']:
        lines.append(f"\n--- {day['day'].upper()} (Total: ~{day['total_calories']} kcal | {day['total_protein']}g Protein) ---")
        for m in day['meals']:
            lines.append(f"  [{m['meal_type']}] {m['name']}")
            lines.append(f"    Portion: {m['portion']}")
            lines.append(f"    Calories: {m['calories']} kcal | Protein: {m['protein_g']}g | Carbs: {m['carbs_g']}g | Fat: {m['fat_g']}g")
            if m.get('allergens') and m['allergens'] != 'none':
                lines.append(f"    Allergens: {m['allergens']}")

    lines.append("\n\n[4] WEEKLY WORKOUT SCHEDULE (ML Ranked)")
    lines.append("=" * 70)
    for wday in plan['workout_plan']:
        lines.append(f"\n--- {wday['day'].upper()}: {wday['focus']} ---")
        if wday['is_rest']:
            lines.append(f"  Status: REST / ACTIVE RECOVERY")
            lines.append(f"  Notes:  {wday['notes']}")
        else:
            lines.append(f"  Notes:  {wday['notes']}")
            for idx, ex in enumerate(wday['exercises'], 1):
                lines.append(f"  {idx}. {ex['name']} (Target: {ex['target_muscle']} | Equip: {ex['equipment']})")
                lines.append(f"     Sets & Reps: {ex['sets']} sets x {ex['reps']} | Rest: {ex['rest']}")
                lines.append(f"     Form Cue:    {ex['form_cues']}")
                lines.append(f"     Home Alt:    {ex['home_alternative']}")

    lines.append("\n" + "=" * 70)
    lines.append("DISCLAIMER: Educational mini-project demonstration only. Not medical advice.")
    lines.append("=" * 70)

    text_output = "\n".join(lines)
    filename = f"Diet_and_Workout_Plan_#{plan['id']}.txt"
    return Response(
        text_output,
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )


@app.route('/delete/<int:plan_id>', methods=['POST'])
def delete_plan(plan_id):
    """Delete a saved plan."""
    database.delete_plan_by_id(plan_id)
    flash(f"Plan #{plan_id} has been deleted.", "info")
    return redirect(url_for('saved_plans'))


if __name__ == '__main__':
    app.run(debug=True, port=5000)
