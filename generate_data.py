import numpy as np
import pandas as pd
import os


def generate_loan_data(n_samples=10000, random_state=42):
    """
    Generate synthetic loan data for credit risk modeling.
    
    Each feature uses a statistical distribution that matches real-world patterns. 
    Credit scores follow a normal distribution centred around 680. 
    Income is log-normal (most people earn moderate amounts, a few earn a lot). 
    Loan purposes have weighted probabilities matching actual lending data.

    The .clip() calls enforce realistic bounds. Nobody has a credit score of 1500 or negative income.
        
    """
    rng = np.random.default_rng(random_state)

    # Core financial features with realistic distributions
    """
    
    Credit Score: 
    credit_score = rng.normal(680, 80, n_samples).clip(300, 850).astype(int) 
    Generates a normal distribution (a bell curve) centered around an average score of 680 with a standard deviation of 80. The .clip(300, 850) forces the scores to stay within real-world limits (300 to 850), and .astype(int) rounds them to whole numbers.



    Annual Income: 
    annual_income = rng.lognormal(10.8, 0.5, n_samples).clip(20000, 500000) 
    Uses a lognormal distribution. In the real world, most people earn moderate salaries, while a smaller percentage earn very high incomes. This distribution captures that natural skew, and .clip() caps it between $20,000 and $500,000.



    Loan Amount: 
    loan_amount = rng.lognormal(9.5, 0.7, n_samples).clip(1000, 100000) 
    Also uses a lognormal distribution to model loan sizes—meaning most loans are moderate (e.g., $5,000 to $15,000), but a few reach up to a maximum limit of $100,000.



    Debt-to-Income (DTI) Ratio: 
    debt_to_income = rng.uniform(0, 50, n_samples) 
    Generates a flat, uniform distribution, meaning any percentage from 0% to 50% is equally likely to be assigned.



    Employment Length: 
    employment_length = rng.exponential(5, n_samples).clip(0, 40).astype(int) 
    Uses an exponential distribution. Many people have short job tenures (0 to 5 years), and the frequency decreases rapidly as tenure length increases up to 40 years.



    Home Ownership: 
    home_ownership = rng.choice(["RENT", "OWN", "MORTGAGE"], n_samples, p=[0.4, 0.1, 0.5]) 
    Randomly selects home statuses based on specified probabilities (p): 40% renting, 10% owning outright, and 50% paying off a mortgage.



    Loan Purpose: 
    loan_purpose = rng.choice(["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "other"], n_samples, p=[0.4, 0.25, 0.15, 0.1, 0.1]) 
    Categorical generator that weights common borrowing reasons (like 40% for debt consolidation and 25% for credit card refinancing) to mimic realistic banking portfolios.



    Interest Rate: 
    interest_rate = rng.uniform(5, 25, n_samples) 
    Spreads borrowing rates uniformly from a low of 5% to a high of 25%.



    Open Accounts & Historical Delinquencies: 
    open_accounts = rng.poisson(10, n_samples) 
    delinquencies_2yr = rng.poisson(0.3, n_samples) 
    Uses a Poisson distribution, which is ideal for modeling count events. Borrowers will have an average of 10 open accounts, and a very low default-prone average of 0.3 delinquencies in the past 2 years (meaning most values will be 0, some 1, and rarely higher).
    
    """
    
    credit_score = rng.normal(680, 80, n_samples).clip(300, 850).astype(int) # Generates a normal distribution (a bell curve) centered around an average score of 680 with a standard deviation of 80. The .clip(300, 850) forces the scores to stay within real-world limits (300 to 850), and .astype(int) rounds them to whole numbers.
    annual_income = rng.lognormal(10.8, 0.5, n_samples).clip(20000, 500000)
    loan_amount = rng.lognormal(9.5, 0.7, n_samples).clip(1000, 100000)
    debt_to_income = rng.uniform(0, 50, n_samples)
    employment_length = rng.exponential(5, n_samples).clip(0, 40).astype(int)
    home_ownership = rng.choice(
        ["RENT", "OWN", "MORTGAGE"], n_samples, p=[0.4, 0.1, 0.5]
    )
    loan_purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "other"],
        n_samples,
        p=[0.4, 0.25, 0.15, 0.1, 0.1],
    )
    interest_rate = rng.uniform(5, 25, n_samples)
    open_accounts = rng.poisson(10, n_samples)
    delinquencies_2yr = rng.poisson(0.3, n_samples)
    
    
    # Calculate default probability based on feature relationships
    
    """
    The formula combines features into a linear score, then squashes it through a sigmoid function to produce a probability between 0 and 1. Higher credit scores reduce default risk. Higher debt-to-income and delinquencies increase it.

    The result is approximately 20% of records defaulting. This intentional class imbalance mirrors real lending data and will expose a critical problem with naive models later.
    
    The Linear Score: 
    This formula models raw default risk. Positive coefficients (like + 0.02 * debt_to_income or + 0.1 * delinquencies_2yr) increase risk. Negative coefficients (like - 0.005 * credit_score) decrease risk.



    Sigmoid Squashing: 
    default_prob = 1 / (1 + np.exp(-default_prob)) 
    Because the raw linear value can range wildly, this line squashes that value through a mathematical sigmoid function to compress it strictly between 0 and 1. This converts the linear score into a valid probability percentage.



    Adding Messy Real-World Noise: 
    default_prob = (default_prob + rng.uniform(0, 0.1, n_samples)).clip(0, 1) 
    Real-world data is never mathematically perfect. Adding 0% to 10% random noise ensures our synthetic model isn't "perfectly linear," giving your machine learning classifier a more realistic challenge.



    Generating the Binary target: 
    loan_default = (rng.random(n_samples) < default_prob).astype(int) 
    This generates a random float between 0 and 1 for each borrower. If that random decimal is less than the calculated default probability, they default (1). Otherwise, they pay on time (0).
    
    """
    default_prob = (
        -0.005 * credit_score
        + 0.02 * debt_to_income
        + 0.03 * interest_rate
        - 0.000005 * annual_income
        + 0.1 * delinquencies_2yr
        - 0.02 * employment_length
        + 1
    )
    default_prob = 1 / (1 + np.exp(-default_prob))
    default_prob = (default_prob + rng.uniform(0, 0.1, n_samples)).clip(0, 1)
    loan_default = (rng.random(n_samples) < default_prob).astype(int)

    df = pd.DataFrame({
        "credit_score": credit_score,
        "annual_income": np.round(annual_income, 2),
        "loan_amount": np.round(loan_amount, 2),
        "debt_to_income": np.round(debt_to_income, 2),
        "employment_length": employment_length,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "interest_rate": np.round(interest_rate, 2),
        "open_accounts": open_accounts,
        "delinquencies_2yr": delinquencies_2yr,
        "loan_default": loan_default,
    })

    return df




if __name__ == "__main__":
    
    
    """
    
    The if __name__ == "__main__" block runs only when you execute the file directly. It creates the data directory, generates 10,000 loan records, saves them to data/loans.csv, and prints summary statistics so you can confirm it worked.
    
    """
    
    os.makedirs("data", exist_ok=True)
    df = generate_loan_data()
    df.to_csv("data/loans.csv", index=False)
    print(f"Generated {len(df)} loan records")
    print(f"Default rate: {df['loan_default'].mean():.1%}")
    print(f"\nSample records:")
    print(df.head())