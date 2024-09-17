from pathlib import Path
import os
import click
from dotenv import find_dotenv, load_dotenv, set_key
import webbrowser
import time 

@click.command()
@click.argument('root_dir', type=click.Path(), required=False)
def main(root_dir=None):
    """
    Sets up the project environment by creating a root directory and setting environment variables.
    This function performs the following steps:
    
    1. Checks if a .env file exists in the current directory. If not, it creates one.
    2. If `root_dir` is not provided, it sets the current working directory as the root directory.
    3. Creates the root directory if it does not exist.
    4. Sets several environment variables in the .env file, including:
       - ROOT_DIRECTORY: The path to the root directory.
       - MLFLOW_TRACKING_USERNAME: The username for MLflow tracking.
       - MLFLOW_TRACKING_PASSWORD: The password for MLflow tracking.
       - MLFLOW_TRACKING_URI: The URI for MLflow tracking.
       - DAGSHUB_USER_TOKEN: The user token for DagsHub.
    
    Parameters
    ----------
    root_dir : str or None, optional
        The path to the root directory. If None, the current working directory will be used.
    
    Returns
    -------
    None
    """
    env_path = Path('.env')

    if not env_path.exists():
        env_path.touch()
    
    if root_dir is None:
        root_dir = Path(os.getcwd()).resolve()
        print("You haven't provided the path to the root directory. The current directory will be set as the root directory.")
    else:
        root_dir = Path(root_dir).resolve()

    root_dir.mkdir(parents=True, exist_ok=True)
    full_root_dir = root_dir.resolve()
    print(f"Root directory set to: {full_root_dir}")
    
    set_key(env_path, 'ROOT_DIRECTORY', str(full_root_dir))
    set_key(env_path, 'MLFLOW_TRACKING_USERNAME', 'tnbmarketplace')
    set_key(env_path, 'MLFLOW_TRACKING_PASSWORD', '0d957e7b20c38643e8fd8de6d9d8e1de130caf90')
    set_key(env_path, 'MLFLOW_TRACKING_URI', 'https://dagshub.com/tnbmarketplace/mlflow_experiment_tracking.mlflow')
    set_key(env_path, 'DAGSHUB_USER_TOKEN', 'fc957a0e9846b45be51bcea1a3ea28f7a3f236aa')

    print(f"Project environment variables have been set in {env_path.resolve()}")
    print("now you are redirecting to the getting started manual ...")
    time.sleep(4) 
    getting_started_path = "file:///"/root_dir/'docs'/'_build'/'html'/'getting-started.html'
    webbrowser.open(getting_started_path,new=2)

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    main()
