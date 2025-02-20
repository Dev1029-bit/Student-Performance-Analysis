**Student Performance Analysis**
   This project predicts student performance (Pass/Fail) using a Logistic Regression model trained on the Students Performance in Exams dataset.
   The web application is built using Flask and provides an interactive interface for users to input scores and get predictions.

**Features*
   Predicts student performance based on scores in:
        Math
        Reading
        Writing
        Parental Level of Education
        Test Preparation Course
   Interactive web interface using HTML and enhanced CSS styles.
   Model trained using Logistic Regression.
   Dataset imported from Kaggle using kagglehub.

**Tech Stack**
    Backend: Flask, scikit-learn, pandas, numpy
    Frontend: HTML, CSS (Bootstrap for responsive design)
    Model: Logistic Regression
    Deployment: Localhost (Flask)
    
**Clone the Repository:**
    git clone https://github.com/your-username/student-performance-analysis.git
    cd student-performance-analysis
    
**Install Required Packages:**
    pip install -r requirements.txt
    
**Download Dataset: Ensure you have Kaggle API credentials and run:**
    import kagglehub
    path = kagglehub.dataset_download("spscientist/students-performance-in-exams")
    
**Run the Application:**
    python app.py
Open http://localhost:5000 in your browser.




