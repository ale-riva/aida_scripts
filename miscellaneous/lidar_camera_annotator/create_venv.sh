# Create a .sus venv where to run sustechpoints
VENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_NAME=".sus"

echo "creating venv at $VENV_DIR/$VENV_NAME...."
python3 -m venv $VENV_DIR/$VENV_NAME
source $VENV_DIR/$VENV_NAME/bin/activate

echo "Install SUSTechPOINTS requirements...."
cd $VENV_DIR/SUSTechPOINTS
pip install -r requirement.txt

echo "environment created"