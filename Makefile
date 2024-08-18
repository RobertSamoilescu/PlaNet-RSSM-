make-install:
	poetry install

fmt:
	black planet scripts

lint:
	black --check planet scripts
	flake8 planet scripts --ignore=E402
	mypy planet scripts