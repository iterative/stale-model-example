import os
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from dotenv import load_dotenv
import bike_predictor

app = Flask(__name__)

CORS(app)

load_dotenv()
DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING')

app.config[
    "SQLALCHEMY_DATABASE_URI"
] = DB_CONNECTION_STRING

db = SQLAlchemy(app)

migrate = Migrate(app, db)


class EventsModel(db.Model):
    __tablename__ = "events"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String())
    date = db.Column(db.Date())
    time = db.Column(db.Time())
    location = db.Column(db.String())
    description = db.Column(db.String())

    def __init__(self, title, date, time, location, description):
        self.title = title
        self.date = date
        self.time = time
        self.location = location
        self.description = description

    def __repr__(self):
        return f"<Event {self.title}>"


@app.route("/event", methods=["POST"])
def create_event():
    if request.is_json:
        data = request.get_json()

        new_event = EventsModel(
            title=data["title"],
            date=data["date"],
            time=data["time"],
            location=data["location"],
            description=data["description"],
        )

        db.session.add(new_event)
        db.session.commit()

        bike_predictor.predict(new_event)

        return {"message": f"event {new_event.title} has been created successfully."}
    else:
        return {"error": "The request payload is not in JSON format"}


if __name__ == "__main__":
    app.run()