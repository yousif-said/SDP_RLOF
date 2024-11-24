# RLHF Interface

This project is a MVP of a web application that uses GPT-2 to generate text continuations based on prompts from a dataset. Users can label the generated text as toxic or not, and the results are stored in a JSON file. This is the first step in developing a simple RLHF pipeline allowing for easy training of models from human feedback. 

## Authors

This was made by UConn Senior Design Team 7 

## Features

- Display a random prompt from a dataset.
- Generate a text continuation using GPT-2.
- Label the generated text as toxic or not.
- Store the results in a JSON file.

## Requirements

- Python 3.6+
- Flask
- Transformers
- Torch

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/rlhf-interface.git
   cd rlhf-interface
   ```

2. Install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

3. Ensure you have the following files in the `datasets` directory:
   - `dataset.json`
   - `toxic_dataset.json`

## Usage

1. Run the Flask application:

   ```sh
   python app.py
   ```

2. Open your web browser and go to `http://127.0.0.1:5000/`.

3. Use the interface to generate text continuations and label them as toxic or not.

## File Structure

- `app.py`: The main Flask application.
- `templates/index.html`: The HTML template for the web interface.
- `static/js/script.js`: The JavaScript file for handling frontend interactions.
- `datasets/dataset.json`: The dataset of non-toxic prompts.
- `datasets/toxic_dataset.json`: The dataset of toxic prompts.
- `results.json`: The file where labeled results are stored.

## Example

1. The left box displays a random prompt from the selected dataset.
2. Click the "Click Here to Generate a Continuation" button to generate a response.
3. The generated response is displayed in the right box.
4. Label the response as toxic or not using the buttons provided.
5. The results are stored in `results.json`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.
