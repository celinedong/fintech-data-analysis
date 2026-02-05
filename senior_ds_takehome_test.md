# Senior Data Scientist Take-Home Assessment

**Time Allocation:** 3-4 hours  
**Programming Language:** Python  
**Submission Format:** Jupyter notebook (.ipynb) or Python script (.py) with accompanying markdown report

---

## Overview

You've been hired as a senior data scientist at **FinFlow**, a fintech company that provides small business loans. The company wants to improve its loan approval process by building a more sophisticated risk assessment system.

Your task is to analyze historical loan data, build predictive models, and provide actionable insights to the business team.

---

## Business Context

FinFlow currently approves loans based on a simple rule-based system. The company has noticed:
- **High default rates** in certain customer segments
- **Missed opportunities** where good customers were rejected
- **Lack of transparency** in why certain applications are approved/rejected

Your analysis will inform the development of a new data-driven loan approval system.

---

## Dataset

You will work with synthetic loan application data. The dataset contains:

**Features:**
- `application_id`: Unique identifier
- `annual_income`: Annual income in USD
- `credit_score`: Credit score (300-850)
- `loan_amount`: Requested loan amount
- `loan_purpose`: Purpose of loan (business_expansion, equipment, working_capital, real_estate)
- `years_in_business`: Number of years the business has been operating
- `num_employees`: Number of employees
- `previous_loans`: Number of previous loans with FinFlow
- `debt_to_income_ratio`: Current debt payments as % of income
- `industry`: Business industry sector
- `monthly_revenue`: Average monthly revenue
- `has_collateral`: Whether collateral is offered (yes/no)
- `education_level`: Owner's education level
- `geographic_region`: Business location (northeast, southeast, midwest, west, southwest)

**Target Variable:**
- `default`: Whether the loan defaulted (1 = default, 0 = paid in full)

---

## Part 1: Data Exploration & Quality Assessment (30 minutes)

### Tasks:

1. **Load and inspect the data**
   - Generate or load the dataset (code provided below)
   - Examine data types, missing values, and basic statistics
   - Document any data quality issues

2. **Exploratory Data Analysis**
   - Analyze the distribution of the target variable (class imbalance?)
   - Identify key patterns in defaults across different features
   - Create 3-5 visualizations that tell a compelling story about the data
   - Calculate and interpret the current default rate

3. **Feature Analysis**
   - Which features show the strongest relationship with defaults?
   - Are there any concerning correlations or multicollinearity issues?
   - Identify any outliers or anomalies

### Deliverables:
- Summary of data quality findings
- Key visualizations with interpretations
- List of features ranked by importance/relevance

---

## Part 2: Feature Engineering & Preprocessing (30 minutes)

### Tasks:

1. **Create new features** that might improve model performance. Examples:
   - Revenue to loan amount ratio
   - Credit score buckets
   - Loan burden (loan_amount / annual_income)
   - Risk score composites
   - Interaction features
   - Any other domain-relevant features you identify

2. **Handle missing data** (if any)
   - Document your approach and rationale

3. **Encode categorical variables**
   - Choose appropriate encoding methods for each categorical feature
   - Explain your choices

4. **Address class imbalance** (if present)
   - Decide on a strategy and implement it
   - Justify your approach

5. **Create train/test split**
   - Use appropriate methodology
   - Document any temporal considerations

### Deliverables:
- Description of engineered features and their rationale
- Preprocessing pipeline code
- Justification for key decisions

---

## Part 3: Model Development (60-90 minutes)

### Tasks:

1. **Build at least 3 different models**, such as:
   - Logistic Regression (baseline)
   - Random Forest or Gradient Boosting
   - Another model of your choice
   
2. **Model evaluation**
   - Choose appropriate metrics given the business context
   - Use cross-validation
   - Create a comparison table of model performance
   - Analyze where models succeed and fail

3. **Hyperparameter tuning**
   - Perform tuning on your best model(s)
   - Document the search space and final parameters

4. **Feature importance analysis**
   - Identify which features drive predictions
   - Validate that the model is learning sensible patterns

5. **Model interpretation**
   - Provide example predictions with explanations
   - Discuss model transparency and explainability

### Evaluation Metrics to Consider:
- Which metric matters most for this business problem?
- What is the cost of false positives vs false negatives?
- Should we optimize for precision, recall, F1, AUC-ROC, or something else?

### Deliverables:
- Trained models with performance metrics
- Feature importance analysis
- Model comparison and recommendation
- Discussion of metric choice

---

## Part 4: Business Recommendations (30-45 minutes)

### Tasks:

1. **Risk segmentation**
   - Divide applicants into risk tiers (e.g., low, medium, high risk)
   - Provide clear criteria for each tier
   - Recommend different strategies for each tier

2. **Decision threshold analysis**
   - Should the company use a 0.5 probability threshold?
   - Create a threshold analysis showing trade-offs
   - Recommend an optimal threshold with business justification

3. **Actionable insights**
   - What are the top 5 insights from your analysis?
   - Which customer segments should FinFlow target/avoid?
   - What additional data would improve the model?

4. **Implementation considerations**
   - How would you deploy this model?
   - What monitoring would you recommend?
   - How would you handle model drift?

### Deliverables:
- Executive summary (max 1 page)
- Risk segmentation framework
- Top actionable recommendations
- Implementation roadmap

---

## Part 5: Code Quality & Documentation (Throughout)

### Requirements:

- **Clean, readable code** with consistent style
- **Meaningful variable names** and structure
- **Comments** explaining complex logic
- **Functions** for reusable code
- **Error handling** where appropriate
- **Reproducibility**: Set random seeds where needed
- **Documentation**: Markdown cells explaining your thought process

---

## Evaluation Criteria

Your submission will be evaluated on:

1. **Technical Skills (40%)**
   - Correct implementation of ML techniques
   - Appropriate use of Python libraries
   - Statistical rigor
   - Model performance

2. **Problem Solving (25%)**
   - Feature engineering creativity
   - Handling of edge cases
   - Thoughtful approach to challenges

3. **Business Acumen (20%)**
   - Understanding of business context
   - Actionable recommendations
   - Communication of trade-offs

4. **Code Quality (15%)**
   - Readability and organization
   - Documentation
   - Reproducibility
   - Best practices

---

## Data Generation Code

Use this code to generate the synthetic dataset:

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Set random seed for reproducibility
np.random.seed(42)

# Generate base data
n_samples = 5000

# Create features
data = {
    'application_id': range(1, n_samples + 1),
    'annual_income': np.random.gamma(2, 30000, n_samples),
    'credit_score': np.random.beta(8, 2, n_samples) * 550 + 300,
    'loan_amount': np.random.gamma(2, 15000, n_samples),
    'years_in_business': np.random.exponential(3, n_samples),
    'num_employees': np.random.poisson(8, n_samples) + 1,
    'previous_loans': np.random.poisson(1.5, n_samples),
    'debt_to_income_ratio': np.random.beta(2, 5, n_samples),
    'monthly_revenue': np.random.gamma(2, 5000, n_samples),
    'loan_purpose': np.random.choice(
        ['business_expansion', 'equipment', 'working_capital', 'real_estate'],
        n_samples,
        p=[0.35, 0.25, 0.25, 0.15]
    ),
    'industry': np.random.choice(
        ['retail', 'services', 'manufacturing', 'technology', 'healthcare', 'food'],
        n_samples,
        p=[0.20, 0.25, 0.15, 0.15, 0.15, 0.10]
    ),
    'has_collateral': np.random.choice(['yes', 'no'], n_samples, p=[0.4, 0.6]),
    'education_level': np.random.choice(
        ['high_school', 'bachelors', 'masters', 'phd'],
        n_samples,
        p=[0.25, 0.45, 0.25, 0.05]
    ),
    'geographic_region': np.random.choice(
        ['northeast', 'southeast', 'midwest', 'west', 'southwest'],
        n_samples
    )
}

df = pd.DataFrame(data)

# Create realistic target variable based on features
risk_score = (
    -0.3 * (df['credit_score'] - 600) / 100 +
    0.4 * df['debt_to_income_ratio'] * 10 +
    -0.2 * np.log1p(df['annual_income']) +
    0.3 * (df['loan_amount'] / df['annual_income']) * 10 +
    -0.15 * np.minimum(df['years_in_business'], 10) +
    (df['has_collateral'] == 'no') * 0.8 +
    (df['previous_loans'] > 2) * -0.5 +
    np.random.normal(0, 2, n_samples)
)

# Convert to probability and then to binary outcome
default_prob = 1 / (1 + np.exp(-risk_score))
df['default'] = (default_prob > np.random.uniform(0, 1, n_samples)).astype(int)

# Add some missing values (realistic scenario)
missing_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
df.loc[missing_indices, 'monthly_revenue'] = np.nan

missing_indices = np.random.choice(n_samples, size=int(0.03 * n_samples), replace=False)
df.loc[missing_indices, 'credit_score'] = np.nan

# Save dataset
df.to_csv('loan_data.csv', index=False)
print(f"Dataset created with {len(df)} samples")
print(f"Default rate: {df['default'].mean():.2%}")
```

---

## Submission Instructions

Please submit:

1. **Jupyter notebook** (.ipynb) OR **Python script** (.py) with your complete analysis
2. **README.md** with:
   - Environment setup instructions (libraries and versions)
   - How to run your code
   - Summary of your approach
3. **executive_summary.pdf** (optional but recommended)
4. Any additional files (saved models, additional visualizations, etc.)

**Submission deadline:** [Insert deadline]

**Submit to:** [Insert email/portal]

---

## Questions?

If you have clarifying questions about the assignment, please email [contact email]. We expect senior candidates to make reasonable assumptions where information is ambiguous, but we're happy to clarify business requirements or technical constraints.

---

## Tips for Success

- **Start with the data**: Understanding the data deeply will guide everything else
- **Think like a business partner**: Your recommendations should be actionable
- **Show your work**: We want to see your thought process, not just the final answer
- **Be honest about limitations**: Acknowledging trade-offs shows maturity
- **Quality over quantity**: A focused, well-executed analysis beats a sprawling mediocre one
- **Test your code**: Make sure everything runs without errors

Good luck! We're excited to see your approach to this problem.
