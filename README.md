# Space debris removal model published in Frontiers

Space debris removal model based on PyKEP scientific library. Developed within ESA Ariadna study project (University of Liverpool and European Space Agency cooperation).  

Authors: Richard Klima, Daan Bloembergen, Rahul Savani, Karl Tuyls, Alexander Wittig, Andrei Sapera and Dario Izzo.  
Corresponding author: Richard Klima - richard.klima(at)liverpool(dot)ac(dot)uk  

This is a significantly improved model based on our previous work (Klima, R., Bloembergen, D., Savani, R., Tuyls, K., Hennes, D., and Izzo, D. (2016a). Space debris removal:963A game theoretic analysis, Games7, 20).

There is a break-up model (breakup.py) and collision model (cubeAlone.py) which is based on CUBE method for evaluating probability of collision of two objects.  

You can run the space debris simulator by running the main class main.py. To run this model successfully you will need to install PyKEP scientific library available at https://esa.github.io/pykep/.  

You can run the approximation (surrogate) model by running simModel.py, where you can for example use Q-learning to learn multi-agent strategies.


The initial setting is:  

time horizon - 100 years  

SATCAT - satellite catalogue  
TLE - two-line element set database  

