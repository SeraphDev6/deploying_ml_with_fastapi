from typing import List
from pydantic import BaseModel, Field, validator
from enum import Enum
from pandas import read_json
from math import isnan


class Salary(str, Enum):
    Under_50k = "<=50K"
    Over_50k = ">50K"


class Output(BaseModel):
    actual: Salary | None
    predicted: Salary
    correct: bool | float | None

    @validator("correct")
    def map_nan_to_none(cls, value):
        if isnan(value):
            return None
        return value


class ListOutput(BaseModel):
    results: List[Output]


class Workclass(str, Enum):
    State_gov = "State-gov"
    Self_emp_not_inc = "Self-emp-not-inc"
    Private = "Private"
    Federal_gov = "Federal-gov"
    Local_gov = "Local-gov"
    Unknown = "?"
    Self_emp_inc = "Self-emp-inc"
    Without_pay = "Without-pay"
    Never_worked = "Never-worked"


class Education(str, Enum):
    Bachelors = "Bachelors"
    HS_grad = "HS-grad"
    eleventh = "11th"
    Masters = "Masters"
    ninth = "9th"
    Some_college = "Some-college"
    Assoc_acdm = "Assoc-acdm"
    Assoc_voc = "Assoc-voc"
    seventh_8th = "7th-8th"
    Doctorate = "Doctorate"
    Prof_school = "Prof-school"
    fifth_6th = "5th-6th"
    tenth = "10th"
    first_4th = "1st-4th"
    Preschool = "Preschool"
    twelth = '12th'


class MaritalStatus(str, Enum):
    Never_married = "Never-married"
    Married_civ_spouse = "Married-civ-spouse"
    Divorced = "Divorced"
    Married_spouse_absent = "Married-spouse-absent"
    Separated = "Separated"
    Married_AF_spouse = "Married-AF-spouse"
    Widowed = 'Widowed'


class Occupation(str, Enum):
    Adm_clerical = "Adm-clerical"
    Exec_managerial = "Exec-managerial"
    Handlers_cleaners = "Handlers-cleaners"
    Prof_specialty = "Prof-specialty"
    Other_service = "Other-service"
    Sales = "Sales"
    Craft_repair = "Craft-repair"
    Transport_moving = "Transport-moving"
    Farming_fishing = "Farming-fishing"
    Machine_op_inspct = "Machine-op-inspct"
    Tech_support = "Tech-support"
    Unknown = "?"
    Protective_serv = "Protective-serv"
    Armed_Forces = "Armed-Forces"
    Priv_house_serv = 'Priv-house-serv'


class Relationship(str, Enum):
    Not_in_family = "Not-in-family"
    Husband = "Husband"
    Wife = "Wife"
    Own_child = "Own-child"
    Unmarried = "Unmarried"
    Other_relative = 'Other-relative'


class Race(str, Enum):
    White = "White"
    Black = "Black"
    Asian_Pac_Islander = "Asian-Pac-Islander"
    Amer_Indian_Eskimo = "Amer-Indian-Eskimo"
    Other = "Other"


class Sex(str, Enum):
    Male = "Male"
    Female = "Female"


class NativeCountry(str, Enum):
    United_States = "United-States"
    Cuba = "Cuba"
    Jamaica = "Jamaica"
    India = "India"
    Unknown = "?"
    Mexico = "Mexico"
    South = "South"
    Puerto_Rico = "Puerto-Rico"
    Honduras = "Honduras"
    England = "England"
    Canada = "Canada"
    Germany = "Germany"
    Iran = "Iran"
    Philippines = "Philippines"
    Italy = "Italy"
    Poland = "Poland"
    Columbia = "Columbia"
    Cambodia = "Cambodia"
    Thailand = "Thailand"
    Ecuador = "Ecuador"
    Laos = "Laos"
    Taiwan = "Taiwan"
    Haiti = "Haiti"
    Portugal = "Portugal"
    Dominican_Republic = "Dominican-Republic"
    El_Salvador = "El-Salvador"
    France = "France"
    Guatemala = "Guatemala"
    China = "China"
    Japan = "Japan"
    Yugoslavia = "Yugoslavia"
    Peru = "Peru"
    Outlying_US = 'Outlying-US(Guam-USVI-etc)'
    Scotland = "Scotland"
    Trinadad_Tobago = 'Trinadad&Tobago'
    Greece = "Greece"
    Nicaragua = "Nicaragua"
    Vietnam = "Vietnam"
    Hong = "Hong"
    Ireland = "Ireland"
    Hungary = "Hungary"
    Holand_Netherlands = 'Holand-Netherlands'


class Input(BaseModel):
    age: int
    workclass: Workclass
    fnlgt: int
    education: Education
    education_num: int = Field(alias="education-num")
    marital_status: MaritalStatus = Field(alias="marital-status")
    occupation: Occupation
    relationship: Relationship
    race: Race
    sex: Sex
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: NativeCountry = Field(alias="native-country")
    salary: Salary | None

    def to_df(self):
        return read_json(f"[{self.json().replace('_', '-')}]")


class MassInput(BaseModel):
    inputs: List[Input]

    def to_df(self):
        i = self.json().replace("_", "-").replace(
            "{\"inputs\": ", "").replace("]}", "]")
        print(i)
        return read_json(i, orient="records")
