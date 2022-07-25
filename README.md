# Weight, gender, age classifier with Flask

## Getting started (using Python virtualenv)
1. Clone the repository.
    ```
    git clone https://github.com/moxi43/weight_gender_age_detection.git
    ```

2. Upload [the model](https://drive.google.com/file/d/1Jsn7IYJ12rzFq8z1i98SfUe8qqLH1uQ6/view) and put it in ./weight_gender_age_detection/models/
    ```
    cd models
    ```
3. Create a virtual environment for the project.

    1. Install `virtualenv`:
        ```
        pip install virtualenv
        ```
    2. Create a Python virtual environment:
        ```
        virtualenv venv
        ```
    3. Activate virtual environment:
        
        1.Windows:
        ```
        cd venv\Scripts
        activate
        cd ..\..
        ```
        2.Linux / Mac:
        ```
        source venv/bin/activate
        ```
4. Install libraries:

    ```
    pip install -r requirements.txt
    ```

### Run the code

* Run the app:
    ```
    cd src
    flask run
    ```
    or
    ```
    python srс/app.py 
    ```
* Run on a specific port:
    ```
    cd src
    flask run -p <port>
    ```
    or
    ```
    python src/app.py -p <port>
    ```

## Built With

* [Pytorch](https://pytorch.org/) - The Machine Learning framework used
* [Flask](http://flask.palletsprojects.com/en/1.1.x/) - The web server library

