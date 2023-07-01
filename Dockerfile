FROM python:3.10-slim-bullseye

ENV PYTHONUNBUFFERED True
# COPY . ./

RUN apt-get -y update
RUN apt-get -y install git
ENV APP_HOME /app
WORKDIR $APP_HOME

RUN git clone https://github.com/SeraphDev6/deploying_ml_with_fastapi.git .
RUN pip install -r requirements.txt
RUN python -m dvc remote modify storage --local gdrive_user_credentials_file ${AUTH_PATH}

RUN python -m dvc pull -r drive

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]