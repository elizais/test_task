FROM tiangolo/uwsgi-nginx-flask:python:3.9

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt

COPY . /app

#EXPOSE 5000

#CMD ["python3", "-m" , "flask", "run.py", "--host=0.0.0.0"]


