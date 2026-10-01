# Deep Learning-Based Satellite Image Change Detection

This project is a Deep Learning Minimum Viable Product (MVP) designed to detect environmental land-cover changes between two satellite images of the same area taken at different times, and subsequently calculate an Environmental Risk Score.

## Viva Explanation: How It Works
The system uses **Transfer Learning** based on a pre-trained **MobileNetV2** model. We replace the final layer and fine-tune the network on the **EuroSAT dataset** (a dataset of satellite images categorized into 10 land-cover classes). Because the input satellite images covering a large area are too large to pass through the model directly, the inference pipeline splits each input image into a grid of smaller square **tiles** (e.g., 64x64 pixels). The neural network classifies each tile independently into one of 10 EuroSAT classes, which we then map into 4 broader categories: Vegetation, Water, Built-up, and Barren. By comparing the tile-by-tile classification map of an older image to a newer image, the system computes percentage changes in land cover. Finally, a rule-based algorithm calculates an **Environmental Risk Score** by assigning penalty points for negative changes, such as a decrease in vegetation or an increase in built-up areas.

## Setup Instructions

1. **Install Dependencies**
   Ensure you have Python installed. Run the following command to install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**
   Before running the dashboard, you must download the dataset and fine-tune the model. This script will automatically download the EuroSAT dataset, train the model, and save the weights to `models/landcover_model.pth`.
   ```bash
   python train_model.py
   ```

3. **Run the Dashboard**
   Once the model is trained, launch the Streamlit web application:
   ```bash
   streamlit run app.py
   ```

## Project Structure
- `app.py`: Streamlit dashboard handling UI, file uploads, and displaying results.
- `train_model.py`: Fine-tunes the MobileNetV2 model and saves it.
- `predict.py`: Handles loading the model and running tile-by-tile inference.
- `change_analysis.py`: Calculates area percentages, the change mask, and the risk score.
- `utils.py`: Helper functions for image loading, resizing, and overlays.
- `config.py`: Configuration settings, class mappings, and risk weights.
