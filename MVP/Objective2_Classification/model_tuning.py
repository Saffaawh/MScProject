
from sklearn.tree import export_graphviz
import graphviz
import joblib # You need to explicitly import joblib to use it

# --- Configuration ---
MODEL_PATH = 'decision_tree_model.joblib'
OUTPUT_FILE_NAME = "decision_tree_visualization"

# --- 1. Load the Trained Model ---
try:
    # Load the Decision Tree model using joblib.
    # NOTE: Ensure you trained and saved a Decision Tree Classifier previously.
    dt_classifier = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please check the path.")
    exit()

# --- 2. Define Features and Classes ---
# You MUST replace this list with the exact feature names used after preprocessing!
# For a basic Decision Tree, these are your column names before OneHotEncoding.
#feature_names = [
  #  'Scan Type_Brain', 'Mobility Needs_Hoist Required', 'Age', 
  #  'Gender_Male', 'Gender_Female', 'Mobility Needs_Wheelchair User', '...'] 
# Note: For classification, you often need the OneHotEncoded feature names. #my wodel has 89 features so getting error when i name features
num_features = dt_classifier.n_features_in_ # Get number of features from the model
feature_names = [f'feature_{i}' for i in range(num_features)] # Placeholder names

print(f"Number of features in the model: {num_features}")

# --- 3. Export the tree structure to a .dot file ---
dot_data = export_graphviz(
    dt_classifier, 
    out_file=None, 
    feature_names=feature_names, 
    class_names=['Class 1', 'Class 2', 'Class 3'], 
    filled=True, 
    rounded=True, 
    special_characters=True,
    max_depth=5 # Set a maximum depth to prevent extremely large images
)

# --- 4. Render the .dot file into a viewable image (PNG or PDF) ---
graph = graphviz.Source(dot_data)

# Save the rendered output file
graph.render(OUTPUT_FILE_NAME, view=False, format='png') 

print(f"Visualisation complete! Decision tree saved as {OUTPUT_FILE_NAME}.png")