**TEKTOK Session Plan — AI & ML Overview + Hands-on**
=====================================================

**1\. Kickoff**
---------------

### **What is AI?**

AI is the field of creating systems that can perform tasks requiring human-like intelligence (language, vision, decisions, predictions).

### **What is ML?**

ML is a subset of AI where systems learn patterns from data instead of relying on manually written rules.

### **Why ML?**

*   Complex problems cannot be solved with fixed rules.
    
*   ML improves with more data.
    
*   ML enables automation (fraud detection, recommendations, personalization, etc.).
    

**2\. Real-World Examples**
---------------------------

*   Loan approval systems
    
*   Recommendation engines
    
*   Spam detection
    
*   Self-driving cars
    
*   Face recognition
    
*   Medical diagnosis
    

**3\. Basic Terms**
-------------------

*   **Dataset:** Collection of data used in ML.
    
*   **Features:** Input variables (e.g., income, credit score).
    
*   **Label:** Output to predict (e.g., loan approved: yes/no).
    

**4\. Introduction to ML**
==========================

**4.1 Types of ML**
-------------------

*   **Supervised Learning:** Data includes labels. (e.g., loan approval, house-price prediction)
    
*   **Unsupervised Learning:** No labels. (e.g., customer clustering)
    
*   **Reinforcement Learning:** Learn through rewards. (e.g., robotics, game-playing bots)
    

**4.2 What is a Training Dataset?**
-----------------------------------

The portion of the dataset used for the model to learn.Data is normally split into:

*   Training
    
*   Validation
    
*   Test
    

**4.3 ML vs Rule-Based Programming**
------------------------------------

*   Rule-Based: Human writes all rules.
    
*   ML: System learns rules from data.
    
*   ML handles complexity and uncertainty better.
    

**4.4 ML Workflow**
-------------------

1.  **Train** – Model learns patterns from data.
    
2.  **Evaluate** – Test model performance on new data.
    
3.  **Deploy** – Use model in real applications.
    

**5\. Math (Intuition Only but is a must please)**
=============================

**Differential Calculus**
-------------------------

*   Focuses on _rate of change_.
    
*   Used in ML for gradient descent (optimizing model weights).
    

**Integral Calculus**
---------------------

*   Focuses on _accumulation_ and areas under curves.
    
*   Helps in probability and distribution-related concepts.
    

**Underfitting & Overfitting**
------------------------------

*   **Underfitting:** Model too simple, poor performance overall.
    
*   **Overfitting:** Model memorizes training data, performs poorly on new data.
    
*   Goal: **Generalization**.
    

**6\. Hands-On: Loan Approval Prediction**
==========================================

### **Objective**

Build a supervised ML model that predicts if a loan should be approved.

### **Process**

1.  Load and clean dataset
    
2.  Encode categorical features
    
3.  Split into train/test
    
4.  Train classification model
    
    *   Logistic Regression
        
    *   Decision Tree
        
    *   Random Forest
        
    *   XGBoost
        
5.  Evaluate accuracy
    
6.  Show **feature importance** (which features mattered most)
    

### **Algorithm Type**

Loan approval → **Classification problem**(Does **not** require neural networks. Tree-based models often perform best for tabular data.)
