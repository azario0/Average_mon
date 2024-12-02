from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import IntegerField, FloatField, SelectField, SubmitField
from wtforms.validators import DataRequired
import pandas as pd
import joblib

# Define the form class
class PredictionForm(FlaskForm):
    year = IntegerField('Year', validators=[DataRequired()])
    cost_of_living = FloatField('Cost of Living', validators=[DataRequired()])
    housing_cost_percentage = FloatField('Housing Cost Percentage', validators=[DataRequired()])
    tax_rate = FloatField('Tax Rate', validators=[DataRequired()])
    savings_percentage = FloatField('Savings Percentage', validators=[DataRequired()])
    healthcare_cost_percentage = FloatField('Healthcare Cost Percentage', validators=[DataRequired()])
    education_cost_percentage = FloatField('Education Cost Percentage', validators=[DataRequired()])
    transportation_cost_percentage = FloatField('Transportation Cost Percentage', validators=[DataRequired()])
    region = SelectField('Region', validators=[DataRequired()])
    country = SelectField('Country', validators=[DataRequired()])
    submit = SubmitField('Predict')

    def __init__(self, *args, region_choices=[], country_choices=[], **kwargs):
        super(PredictionForm, self).__init__(*args, **kwargs)
        self.region.choices = region_choices
        self.country.choices = country_choices

# Create the Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with your actual secret key

# Load the dataset and unique values
df = pd.read_csv('Cost_of_Living_and_Income_Extended.csv')
regions = df['Region'].unique().tolist()
countries = df['Country'].unique().tolist()

# Load the model
model = joblib.load('best_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = PredictionForm(region_choices=regions, country_choices=countries)
    prediction = None
    if form.validate_on_submit():
        # Collect form data
        data = {
            'Year': [form.year.data],
            'Cost_of_Living': [form.cost_of_living.data],
            'Housing_Cost_Percentage': [form.housing_cost_percentage.data],
            'Tax_Rate': [form.tax_rate.data],
            'Savings_Percentage': [form.savings_percentage.data],
            'Healthcare_Cost_Percentage': [form.healthcare_cost_percentage.data],
            'Education_Cost_Percentage': [form.education_cost_percentage.data],
            'Transportation_Cost_Percentage': [form.transportation_cost_percentage.data],
            'Region': [form.region.data],
            'Country': [form.country.data]
        }
        # Create DataFrame
        input_df = pd.DataFrame(data)
        # Make prediction
        try:
            prediction = round(model.predict(input_df)[0], 2)
        except Exception as e:
            return f"An error occurred: {str(e)}"
    return render_template('index.html', form=form, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)