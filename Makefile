PYTHON= python3

MANAGE= venv/bin/python manage.py

ACTIVATE=. venv/bin/activate

# Django Configrations

PORT= 8003

virtualenv:

		@echo "-> Making Virtual Environment"		@${PYTHON} -m venv venv

install: virtualenv

	@echo "->Activating VirtualENV and then Installing Dependencies mentioned"

	@${ACTIVATE} pip3 install -r requirements.txt

migrate:

	@${MANAGE} makemigrations

	@echo "-> Apply database migrations"

	@${MANAGE} migrate

run: 

	@${MANAGE} runserver ${PORT}
cs:
	@${MANAGE} collectstatic
flush:

	@echo"-> Flushing database"

	@${MANAGE} flush

superuser:

	@${MANAGE} createsuperuser
