from flask import Flask
from flask import request

#add rebeccas dependencies
#add a dummy json object
#try dummy import using posting
#add axios call in backend 

app = Flask(__name__)

# Add app.route methods = POST
@app.route("/")
def hello():
  return "Hello World"
  
#app .run
if __name__ == "__main__":
  app.run()


