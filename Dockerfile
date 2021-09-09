FROM python:3.7-slim

WORKDIR /usr/src/app

RUN pip install torch flask numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

CMD [ "python", "-u", "/usr/src/app/index.py" ]
