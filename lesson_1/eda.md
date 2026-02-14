# Exploratory Data Analysis (EDA) – A Practical Guide for Interns

## 1. What Is Exploratory Data Analysis?

Exploratory Data Analysis (EDA) is the process of **looking at your data before you try to model it or make decisions with it**.

You use EDA to:

- Understand what is in the dataset
- Check data quality (missing values, errors, weird values)
- Spot patterns, trends, and relationships
- Generate ideas and questions for further analysis or modeling

Think of EDA as **getting to know your data**. Before you trust any dashboards, machine learning models, or reports, you must first understand and validate the data.

***

## 2. Why Is EDA Important?

EDA is critical because it:

- **Prevents bad decisions**
    - If the data is wrong or incomplete, your conclusions will be wrong.
- **Improves model performance**
    - Understanding distributions, outliers, and correlations helps you choose better features and algorithms.
- **Guides the rest of the analysis**
    - EDA often reveals what questions are worth asking and which are dead ends.
- **Builds domain understanding**
    - You learn how the business or system behaves through its data.

In practice, **good EDA often takes more time than modeling**, and that is normal and healthy.

***

## 3. Typical EDA Workflow (High-Level)

Here is a common EDA workflow you can follow:

1. **Understand the problem and context**
2. **Load and inspect the raw data**
3. **Check structure and data types**
4. **Check for missing values and duplicates**
5. **Univariate analysis** (each variable on its own)
6. **Bivariate/multivariate analysis** (relationships between variables)
7. **Check for outliers and anomalies**
8. **Summarize findings and next steps**

The sections below walk through these steps in more detail with concrete examples.

***

## 4. Step 1 – Understand the Problem and Context

Before touching the data:

- Clarify the **business question** or **analysis goal**.
    - Examples:
        - “Which customers are most likely to churn?”
        - “What factors drive energy consumption?”
        - “How have sales changed over time by region?”
- Identify:
    - **What is a row?** (e.g., one transaction, one user, one sensor reading)
    - **What are the key variables?** (e.g., date, user_id, amount)
    - **What is the target (if any)?** (e.g., churn flag, revenue, demand)

Write this down. Even a short paragraph helps anchor your EDA.

***

## 5. Step 2 – Load and Inspect the Raw Data

Assuming a typical Python stack (pandas, Jupyter Notebook):

```python
import pandas as pd

df = pd.read_csv("data.csv")  # or another source

# Look at the first few rows
df.head()
```

Things to do immediately:

- **Preview a small sample**
    - `df.head()` – first 5 rows
    - `df.tail()` – last 5 rows
- **Get basic info**
    - `df.shape` – number of rows and columns
    - `df.columns` – column names
    - `df.info()` – data types and non-null counts

Questions to ask yourself:

- Does the row count look reasonable?
- Do the column names make sense?
- Are there obvious issues (e.g., dates stored as strings, numerical fields as text)?

***

## 6. Step 3 – Check Structure and Data Types

Correct data types are essential for accurate analysis.

Useful commands:

```python
df.info()
df.dtypes
```

Common issues to look for:

- **Numbers stored as text**
    - Example: `"123"` instead of `123` → convert with `pd.to_numeric`.
- **Dates stored as text**
    - Convert with `pd.to_datetime`.
- **Categorical fields as text**
    - Consider converting to `category` type if appropriate.

Example conversions:

```python
# Convert a column to numeric
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

# Convert a column to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
```


***

## 7. Step 4 – Check Missing Values and Duplicates

### 7.1 Missing Values

First, see how much data is missing:

```python
df.isna().sum()
df.isna().mean()  # proportion of missing per column
```

Questions:

- Which columns have the most missing values?
- Are missing values random, or do they follow a pattern?

Common approaches:

- **Drop rows** if only a small portion is missing and the row is not critical.
- **Impute values** (fill in):
    - Numerical: mean, median, or a specific value.
    - Categorical: mode (most frequent value) or `"Unknown"`.
- **Leave missing as-is** if the modeling method can handle it or if missingness itself carries meaning.

Example:

```python
# Drop rows with any missing values in important columns
df = df.dropna(subset=["target", "feature_1"])

# Fill numerical column with median
df["age"] = df["age"].fillna(df["age"].median())

# Fill categorical column with 'Unknown'
df["city"] = df["city"].fillna("Unknown")
```


### 7.2 Duplicates

Check for duplicates:

```python
df.duplicated().sum()
```

Remove if necessary:

```python
df = df.drop_duplicates()
```


***

## 8. Step 5 – Univariate Analysis (One Variable at a Time)

Univariate analysis helps you understand the **distribution and behavior** of each variable individually.

### 8.1 Numerical Variables

Use summary statistics:

```python
df["amount"].describe()
```

This gives:

- count, mean, std (standard deviation)
- min, 25%, 50%, 75%, max

Questions:

- Is the distribution symmetric or skewed?
- Are there extremely large or small values?
- Are there impossible values (e.g., negative age)?

Visualizations (using matplotlib / seaborn):

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram
sns.histplot(df["amount"], bins=30, kde=True)
plt.title("Distribution of Amount")

# Boxplot
sns.boxplot(x=df["amount"])
plt.title("Boxplot of Amount")
```


### 8.2 Categorical Variables

Use value counts:

```python
df["city"].value_counts()
df["city"].value_counts(normalize=True)  # proportions
```

Visualizations:

```python
sns.countplot(x="city", data=df)
plt.xticks(rotation=45)
plt.title("Counts by City")
```

Questions:

- Which categories are most common?
- Are there categories with very few observations?
- Do category names need cleaning (e.g., `"NY"`, `"New York"`, `"new york"`)?

***

## 9. Step 6 – Bivariate and Multivariate Analysis

Now look at **relationships between variables**.

### 9.1 Numerical vs Numerical

Use scatter plots and correlation:

```python
# Scatter plot
sns.scatterplot(x="temperature", y="energy_usage", data=df)
plt.title("Energy Usage vs Temperature")

# Correlation matrix
corr = df[["feature1", "feature2", "feature3"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
```

Questions:

- Do you see linear or non-linear relationships?
- Are there clear clusters?
- Are there extreme outliers?


### 9.2 Numerical vs Categorical

Examples:

- Average value by category
- Boxplots by category

```python
# Grouped summary
df.groupby("city")["amount"].describe()

# Boxplot
sns.boxplot(x="city", y="amount", data=df)
plt.xticks(rotation=45)
plt.title("Amount by City")
```

Questions:

- How does the numerical variable differ across groups?
- Are some categories much higher/lower than others?


### 9.3 Categorical vs Categorical

Use cross-tabulations:

```python
pd.crosstab(df["user_type"], df["churned"])
pd.crosstab(df["user_type"], df["churned"], normalize="index")
```

Possible visualizations:

- Stacked bar charts
- Heatmaps

***

## 10. Step 7 – Detecting Outliers and Anomalies

Outliers are values that are **unusually large or small** compared to the rest of the data.

Ways to detect:

- Visual:
    - Boxplots
    - Histograms
    - Scatter plots
- Statistical:
    - Use interquartile range (IQR): values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR

Example (IQR method):

```python
q1 = df["amount"].quantile(0.25)
q3 = df["amount"].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[(df["amount"] < lower_bound) | (df["amount"] > upper_bound)]
```

What to do with outliers?

- **Investigate first** – they might be:
    - Data entry errors
    - Rare but valid events (e.g., very high purchases)
- Possible actions:
    - Correct obvious data errors
    - Remove if they clearly do not make sense
    - Winsorize (cap a variable at a certain percentile)
    - Keep them but use robust models/metrics

***

## 11. Step 8 – Document and Summarize Your Findings

EDA is not just code and plots; it must **end with a clear summary**.

Your summary should include:

- **Data overview**
    - Number of rows and columns
    - Time period covered
    - Key columns and their meaning
- **Data quality issues**
    - Missing values, duplicates
    - Suspicious or impossible values
- **Main patterns and relationships**
    - Which variables are strongly related to the target?
    - Any surprising trends?
- **Outliers and anomalies**
    - What you found and how you handled them
- **Limitations**
    - What cannot be answered from this dataset
    - Any biases or gaps (e.g., missing segments, incomplete history)
- **Next steps**
    - Further analysis to do
    - Features to engineer
    - Questions for domain experts

This summary can be in:

- A section at the bottom of your notebook
- A short Markdown report
- A slide deck for stakeholders

***

## 12. Practical EDA Checklist for Interns

When you get a new dataset, you can use this checklist:

1. **Understand context**
    - [ ] What question are we trying to answer?
    - [ ] What does one row represent?
    - [ ] What are the key columns and target (if any)?
2. **Load and inspect data**
    - [ ] Load into pandas (or another tool)
    - [ ] Look at `head()`, `shape`, `columns`, `info()`
3. **Check types and basic structure**
    - [ ] Data types correct? (numeric, datetime, categorical)
    - [ ] Convert columns where needed
4. **Missing values and duplicates**
    - [ ] `isna().sum()` to see missing values
    - [ ] Decide how to handle missing for each important column
    - [ ] Check for and drop duplicates if necessary
5. **Univariate analysis**
    - [ ] Summary stats for numeric features (`describe()`)
    - [ ] Histograms and boxplots for key numeric features
    - [ ] Value counts and bar charts for key categorical features
6. **Bivariate/multivariate analysis**
    - [ ] Scatter plots for numeric vs numeric
    - [ ] Boxplots or groupby summaries for numeric vs category
    - [ ] Crosstabs for category vs category
    - [ ] Correlation matrix for important numeric variables
7. **Outliers**
    - [ ] Identify unusual values using boxplots or IQR
    - [ ] Decide whether to keep, correct, or remove
8. **Summary**
    - [ ] Write a short narrative summary of what you found
    - [ ] List potential next steps and open questions

***

## 13. Tips and Mindset for Good EDA

- **Be curious**
Treat EDA like investigation, not a checklist you rush through.
- **Move between code and plots**
Use both numbers (summary stats) and visuals (plots) to understand the data.
- **Iterate**
As you learn more, refine your questions and dig deeper into interesting areas.
- **Stay organized**
Structure your notebook with headings and comments so others can follow your work.
- **Explain in plain language**
When summarizing, write so that a non-technical stakeholder can understand the main points.

***

## 14. Minimal Example EDA Template (Python)

You can use this as a starting notebook structure:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("data.csv")

# 2. Basic info
print(df.shape)
print(df.info())
display(df.head())

# 3. Missing values
print(df.isna().sum())

# 4. Summary stats for numeric columns
display(df.describe())

# 5. Example histogram
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in numeric_cols:
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# 6. Correlation heatmap
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
```

Interns can adapt this to each dataset and build more detailed analysis on top.

***

**Summary:**
Exploratory Data Analysis is the **foundation** of any serious analytics or data science work. It helps you understand the dataset, trust the data, and focus your efforts on the most important questions. Use this guide as a step-by-step reference when you start working with any new dataset in the company.
