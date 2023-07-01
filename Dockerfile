FROM python:3.10-slim-bullseye

ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN apt-get -y update
RUN apt-get -y install git
RUN git init

RUN pip install -r requirements.txt
RUN --mount=type=secret,id=_env,dst=/etc/secrets/google_permissions.json cat /etc/secrets/google_permissions.json
RUN python -m dvc pull -r drive
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]