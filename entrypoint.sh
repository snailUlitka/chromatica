#!/bin/sh
set -e

alembic upgrade head
python -m chromatica.api.bootstrap
exec "$@"
