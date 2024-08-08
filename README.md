# road-condition-assessment

## Samples
* Sample images are located in the `samples` directory

## Running the program
* The streamlit app is located at: https://road-condition-assessment.streamlit.app/
  The version on the streamlit servers sometimes stops using opencv headless and completely borks itself

## Local installation

* Clone the repository
* Create a conda environment
* Activate the conda environment
* Run the following commands:
    * `pip install ultralytics`
    * `pip install streamlit streamlit_image_comparison`
    * `pip install -U scikit-image imagecodecs`
* Run the program: `streamlit run main.py`
