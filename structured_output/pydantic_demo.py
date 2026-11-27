from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
  name : str = "abdullah"
  age : Optional[int] = None
  email: EmailStr
  cgpa: float = Field(gt=0, lt=4, default=3, description="A decimal value representing the cgpa of the student")



new_student = {
  "name" : "Masum",
  'email' : 'abc@gmail.com',
  'age' : 23
  }

student = Student(**new_student)
  

print(dict(student)['name'])
