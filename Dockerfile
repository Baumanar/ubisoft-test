FROM python:3.6

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app
RUN ls
ENTRYPOINT [ "python" ]

CMD [ "app.py" ]
