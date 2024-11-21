# Step 1: Create virtual environment
python3 -m venv myenv

# Step 2: Activate virtual environment
source myenv/bin/activate  # macOS/Linux
# OR
myenv\Scripts\activate  # Windows

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run your Python script
python excercise_splitter.py

# Step 5: Deactivate when done
deactivate