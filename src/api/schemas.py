from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class PredictionRequest(BaseModel):
    limit_bal: float = Field(..., description="Amount of given credit in NT dollars")
    sex: int = Field(..., description="Gender (1=male, 2=female)")
    education: int = Field(..., description="Education (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)")
    marriage: int = Field(..., description="Marital status (1=married, 2=single, 3=others)")
    age: int = Field(..., description="Age in years")
    pay_1: int = Field(..., description="Repayment status in September 2005")
    pay_2: int = Field(..., description="Repayment status in August 2005")
    pay_3: int = Field(..., description="Repayment status in July 2005")
    pay_4: int = Field(..., description="Repayment status in June 2005")
    pay_5: int = Field(..., description="Repayment status in May 2005")
    pay_6: int = Field(..., description="Repayment status in April 2005")
    bill_amt1: float = Field(..., description="Amount of bill statement in September 2005")
    bill_amt2: float = Field(..., description="Amount of bill statement in August 2005")
    bill_amt3: float = Field(..., description="Amount of bill statement in July 2005")
    bill_amt4: float = Field(..., description="Amount of bill statement in June 2005")
    bill_amt5: float = Field(..., description="Amount of bill statement in May 2005")
    bill_amt6: float = Field(..., description="Amount of bill statement in April 2005")
    pay_amt1: float = Field(..., description="Amount of previous payment in September 2005")
    pay_amt2: float = Field(..., description="Amount of previous payment in August 2005")
    pay_amt3: float = Field(..., description="Amount of previous payment in July 2005")
    pay_amt4: float = Field(..., description="Amount of previous payment in June 2005")
    pay_amt5: float = Field(..., description="Amount of previous payment in May 2005")
    pay_amt6: float = Field(..., description="Amount of previous payment in April 2005")

class PredictionResponse(BaseModel):
    default_probability: float
    is_default: int
    shap_values: Optional[Dict[str, float]] = None
    top_features: Optional[List[str]] = None
    
class HealthCheck(BaseModel):
    status: str
    version: str
