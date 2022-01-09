FROM python:3.8.10

WORKDIR /equation_solver

COPY . /equation_solver

RUN pip3 install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
CMD ["python3", "app.py"]

#EXPOSE 5000
#CMD ["flask", "run", "-h", "0.0.0.0", "-p", "5000"]

