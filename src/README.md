## Source Code
This directory contains all of the source code defining our models.

## Structure of Directory
- `/mains`: This directory contains all the main files used to invoke the different models. All of the code used in this project is run from within this directory.
- `/modules`: This directory contains all the models. The models are defined as modules which can be imported and invoked within different Python scripts. Each of the modules is defined as a standalone class with an additional dictionary defining the default hyperparameters of the model.
- `/utilities`: This directory contains the different utility classes and methods shared by some modules.
