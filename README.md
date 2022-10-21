# MLProject-Neural_Tigers

## Setting up the virtual environment
Make sure you are inside this project's directory. For this project assignment we will be using vanv because 
venv creates virtual environments in the shell that are fresh and sandboxed, with user-installable libraries, and it's multi-python safe.

venv is a package shipped with python 3. If python 3 is install properly, you already have this virtualization functionality: 

Making sure you are in the project directory, install a new virtual environment for use. Further indicate the folder were all dependencies will exist.:
*We name the file containing the dependencies 'venv'*
```bash
python3 -m venv ./venv
```

Now that we have a new virtual environment installed, it needs to be activated. 
```bash
source ./venv/bin/activate
```
*You should now see '(venv)' displayed to the left of your primary prompt string.*

Verify that you are using python3 and pip from 'venv' file *(our virtual machine source)* 
```bash
which python3
```
```bash
which pip
```

Finished developing for this specific project. To exit virtual machine:
```bash
deactivate
```

For any reason if you need to get rid of a virtual machine entirely, just delete the folder created during installation and activation of the virtual machine *('venv')*.

### Install packages your project depends on 
Virtual machine must be active for the following commands to work properly withg virtual machines.

Install packages the command below.
```bash
pip install [PACKAGE]
```

Uninstall packages the command below.
```bash
pip uninstall [PACKAGE]
```

Check packages already installed.
```bash
pip list
```

Port the dependencies to a file called 'requirement.txt' for others to use in their own virtual environment
```bash
pip freeze > requirements.txt
```

### Installing and Setting Up Jupyter Notebook to work within Virtual Environment and to use venv Kernel 
First, make sure your virtual environment is activated for any of the following.

If you already have Jupyter Notebook already installed in your venv, simply run the following to get started developing with it. 
```bash
jupyter notebook
```

*Then run the following command to make venv one of the available kernels to your jupyter notebook. This should allow you to run jupyter notebook completely in the venv when selected. Make sure you do this for every Jupyter Notebook Project as it is stored outside of the venv. You might have multiple kernels with the same "venv" name. Run this command to ensure the correct kernel is being used with the right notebook. You can remove this kernel profile when finished with project as instructed below.*
```bash
ipython kernel install --user --name=venv 
```


If not already installed, install Jupyter Notebook with:

```bash
pip install jupyter notebook
```
Then to add that kernel profile to list of available kernels: 
```bash
ipython kernel install --user --name=venv 
```


*The above installation commands should ensure that the dependenceis and setup needed to run jupyter notebook are installe. Will also ensure that the proper kernels are installed.*

*If there are any errors, attempt to install the following manually*
```bash
pip install ipython 
```
```bash
pip install ipykernel
```
*NOTE: The IPython kernel is the Python execution backend for Jupyter. The Jupyter Notebook ensures that the IPython kernel (ipykernel) is available. However, if you want to use a kernel with a different version of Python, or in a virtualenv or conda environment, you'll need to install that manually (current process). However, this should all be taken care of with the `pip install jupyter notebook` and `ipython kernel install --user --name=venv` command*


Anytime you boot up Jupyter Notebook with the `jupyter notebook` command, make sure to switch the kernel to venv to abosolutely ensure you are using venv to run code. 

*IN ORDER TO REMOVE THE KERNEL PROFILE YOU ADDED TO A DIRECTORY OUTSIDE VENV EXECUTE THE FOLLOWING:*
After you are done with the project and no longer need the kernel you can uninstall it by running the following code:
```bash
jupyter-kernelspec uninstall venv
```

### Setting up your own venv with someone else's packages. 
Virtual environment source directories should be excluded from repositories and the 'requirements.txt' should be used to set up a newly created virtual environment. This is because of errors that can/probably will arise when using someone else's environment. One being the hardcoded paths issues to arise next time commands like 'pip' are run.

First follow the 'Setting up the virtual environment' walkthrough above and get virtual environment 'venv' directory correctly set up. Then with the 'requirements.txt' file you cloned/pulled/retrieved run this command to install the same dependencies. 
```bash
pip install -r requirements.txt
```
*This assures that you have the same packages and can run the code accompanying the 'requirements.txt' file*

##
