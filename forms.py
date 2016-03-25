from flask.ext.wtf import Form 
from wtforms.fields import TextField, TextAreaField, SubmitField, SelectField
from wtforms.validators import Required

class ContactForm(Form):
    job_title = TextField("Job title", [Required("Enter a job title")])
    noocc = TextField("No of occupations")
    occdrop = SelectField("Choose")
    submit = SubmitField("Send")
