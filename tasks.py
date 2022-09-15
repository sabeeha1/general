from celery import Celery
from time import sleep
app = Celery('tasks',broker = 'http://localhost:15672:')
@app.task
def reverse(text):
    sleep(5)
    return text[::-1]

name = reverse("Sabeeha")
print(name)
