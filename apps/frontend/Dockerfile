FROM python:3.11-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*

COPY ./nginx.conf /etc/nginx/sites-enabled/default
COPY ./start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
