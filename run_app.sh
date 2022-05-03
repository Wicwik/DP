screen -ls | grep -i detached | cut -d. -f1 | awk '{print $1}' | xargs kill &> /dev/null

export FLASK_APP=application
screen -S flask_app -d -m flask run --host=0.0.0.0 --port=8002
