# Cellular Automata learn to distribute information.

In this work we create agents that solely with local information can coordinate together to collectively agree on an image classification. Each agent observes one pixel of the image and its immediate neighbours.

This project is structured as follows:

### Cellular_Automata.ipynb
This is the notebook I use when working on a new task. It has many options that are useful when trying out new things. <span style="color:blue">Ideal if you want to fully customize this code for a new task.</span> 

### Cellular_Automata_task.ipynb
where task is either: fruits, count_digits, xor, emotions, frozen_noise

These notebooks contain all the necessary code for fully training the cellular automata for a particular task. 

### ca_models.py
This module contains the keras models, the loss function, etc. Check it out for full network architecture, loss function, training step, etc.

### datasets_library.py
A module for loading the dataset and converting it into numpy arrays of x_train, x_test, y_train, y_test

### ca_visualisation.py
A module for seeing how the learning is progressing and creating the final videos for the Cellular Automata in action
