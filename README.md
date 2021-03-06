# Cellular Automata learn to distribute information.

In this work we create agents that solely with local information can coordinate together to collectively agree on an image classification. Each agent observes one pixel of the image and its immediate neighbours. You can read a [detailed explanation of this work in my personal website](https://nunocalaim.github.io/cellular_automata)

This project is structured as follows:

### Cellular_Automata
This is the notebook I use when working on a new task. It has many options that are useful when trying out new things. **Ideal if you want to fully customize this code for a new task.** 

### Cellular_Automata_Count_Digits
This notebooks contain all the necessary code for fully training the cellular automata for a particular task. **Ideal if you just want to see how the agents were trained and tested for a particular task**

### ca_models.py
This module contains the keras models, the loss function, etc. Check it out for full network architecture, loss function, training step, etc.

### datasets_library.py
A module for loading the dataset and converting it into numpy arrays of x_train, x_test, y_train, y_test

### ca_visualisation.py
A module for seeing how the learning is progressing and creating the final videos for the Cellular Automata in action
